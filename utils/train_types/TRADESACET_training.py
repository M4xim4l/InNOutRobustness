import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.distances as d

from .ACET_training import ACETObjective, ACETTraining
from .CEDA_training import CEDAObjective
from .TRADES_training import TRADESTraining, TRADESLoss
from .train_type import TrainType
from .train_loss import AccuracyConfidenceLogger, DistanceLogger, ConfidenceLogger, SingleValueLogger, NegativeWrapper
from .helpers import interleave_forward, get_distance
import torch.cuda.amp as amp

import math
######################################################
class TRADESACETTraining(TrainType):
    def __init__(self, model, id_attack_config, od_attack_config, optimizer_config, epochs, device, num_classes,
                 id_trades_weight=1, lr_scheduler_config=None, od_trades=False,
                 acet_obj='kl', lam=1., model_config=None,
                 test_epochs=1, verbose=100, saved_model_dir= 'SavedModels', saved_log_dir= 'Logs'):

        #as we use the same batch for trades/clean training, pass false to train_clean
        super().__init__('TRADESACET', model, optimizer_config, epochs, device, num_classes,
                          lr_scheduler_config=lr_scheduler_config,
                          model_config=model_config, test_epochs=test_epochs,
                          verbose=verbose, saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)

        self.id_distance = get_distance(id_attack_config['norm'])
        self.od_distance = get_distance(od_attack_config['norm'])

        #TRADES
        self.id_attack_config = id_attack_config
        self.id_beta = id_trades_weight

        #CEDA/TRADES specifics
        self.lam = lam
        self.od_attack_config = od_attack_config
        self.acet_obj = acet_obj
        self.od_trades = od_trades,

        self.od_iterator = None


    def requires_out_distribution(self):
        return True

    def create_loaders_dict(self, train_loader, test_loader=None, out_distribution_loader=None, out_distribution_test_loader=None, *args, **kwargs):
        train_loaders = {
            'train_loader': train_loader,
            'out_distribution_loader': out_distribution_loader
        }

        test_loaders = {}
        if test_loader is not None:
            test_loaders['test_loader'] = test_loader
        if out_distribution_test_loader is not None:
            test_loaders['out_distribution_test_loader'] = out_distribution_test_loader

        return train_loaders, test_loaders

    def _validate_loaders(self, train_loaders, test_loaders):
        if not 'train_loader' in train_loaders:
            raise ValueError('Train cifar_loader not given')
        if not 'out_distribution_loader' in train_loaders:
            raise ValueError('Out distribution cifar_loader is required for out distribution training')

    @staticmethod
    def create_id_attack_config(eps, steps, stepsize, norm, momentum=0.9, pgd='pgd', normalize_gradient=True, noise=None):
        return TRADESTraining.create_id_attack_config(eps, steps, stepsize, norm, momentum=momentum,
                                                           pgd=pgd, normalize_gradient=normalize_gradient, noise=noise)

    @staticmethod
    def create_od_attack_config(eps, steps, stepsize, norm, momentum=0.9, pgd='pgd', normalize_gradient=True, noise=None):
        return ACETTraining.create_od_attack_config(eps, steps, stepsize, norm, momentum=momentum,
                                                           pgd=pgd, normalize_gradient=normalize_gradient, noise=noise)

    def _get_ID_TRADES_regularizer(self, epoch):
        trades_reg = TRADESLoss(self.model, epoch, self.id_attack_config, self.classes,
                                log_stats=True, name_prefix='ID',)
        return trades_reg

    def _get_ACET_loss(self, epoch):
        train_criterion = ACETObjective(self.model, epoch, self.od_attack_config, self.acet_obj, self.acet_obj,
                                        self.classes, log_stats=True, name_prefix='OD')
        return train_criterion

    def _get_CEDA_loss(self, epoch):
        train_criterion = CEDAObjective(self.acet_obj, self.classes, log_stats=True, name_prefix='OD')
        return train_criterion

    def _get_OD_TRADES_regularizer(self, epoch):
        trades_reg = TRADESLoss(self.model, epoch, self.od_attack_config, self.classes,
                                log_stats=True, name_prefix='OD',)
        return trades_reg


    def test(self, test_loaders, epoch, test_avg_model=False):
        self.model.eval()

        test_loader = test_loaders['test_loader']
        test_set_batches = len(test_loader)
        ce_loss_clean = self._get_clean_criterion(log_stats=True, name_prefix='Clean')
        id_trades_train_criterion = self._get_ID_TRADES_regularizer(epoch)

        losses = [ce_loss_clean, id_trades_train_criterion]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        acc_conf_adv = AccuracyConfidenceLogger(name_prefix='ID')
        distance_adv = DistanceLogger(self.id_distance, name_prefix='ID')
        loggers = [acc_conf_clean, acc_conf_adv, distance_adv]

        self.output_backend.start_epoch_log(test_set_batches)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                adv_samples = id_trades_train_criterion.inner_max(data, target)

                clean_out, adv_out = interleave_forward(self.model, [data, adv_samples])
                trades_target = F.softmax(clean_out, dim=1)

                loss0 = ce_loss_clean(data, clean_out, data, target)
                loss1 = id_trades_train_criterion(adv_samples, adv_out, data, trades_target)

                acc_conf_clean(data, clean_out, data, target)
                acc_conf_adv(adv_samples, adv_out, data, target)
                distance_adv(adv_samples, adv_out, data, target)
                self.output_backend.log_batch_summary(epoch, batch_idx, False, losses=losses, loggers=loggers)

        self.output_backend.end_epoch_write_summary(losses, loggers, epoch, False)


    def _inner_train(self, train_loaders, epoch, log_epoch=None):
        train_loader = train_loaders['train_loader']
        out_distribution_loader = train_loaders['out_distribution_loader']
        self.model.train()

        train_set_batches = self._get_dataloader_length(train_loader, out_distribution_loader=out_distribution_loader)

        # https: // github.com / pytorch / pytorch / issues / 1917  # issuecomment-433698337
        id_iterator = iter(train_loader)
        if self.od_iterator is None:
            self.od_iterator = iter(out_distribution_loader)

        bs = self._get_loader_batchsize(train_loader)
        od_bs = self._get_loader_batchsize(out_distribution_loader)
        if od_bs != bs:
            raise AssertionError('Out distribution and in distribution cifar_loader need to have the same batchsize')

        ce_loss_clean = self._get_clean_criterion(log_stats=True, name_prefix='Clean')
        id_trades_train_criterion = self._get_ID_TRADES_regularizer(epoch)

        if self.od_trades:
            od_clean_criterion = self._get_CEDA_loss(epoch)
            od_criterion = self._get_OD_TRADES_regularizer(epoch)
            losses = [ce_loss_clean, id_trades_train_criterion, od_clean_criterion, od_criterion]
        else:
            od_clean_criterion = None
            od_criterion = self._get_ACET_loss(epoch)
            losses = [ce_loss_clean, od_criterion, id_trades_train_criterion]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        acc_conf_id = AccuracyConfidenceLogger(name_prefix='ID')
        distance_id = DistanceLogger(self.id_distance, name_prefix='ID')
        distance_od = DistanceLogger(self.od_distance, name_prefix='OD')
        confidence_od = ConfidenceLogger(name_prefix='OD')
        total_loss_logger = SingleValueLogger('Loss')
        lr_logger = SingleValueLogger('LR')

        loggers = [total_loss_logger, acc_conf_id, distance_id, acc_conf_clean, confidence_od, distance_od, lr_logger]

        self.output_backend.start_epoch_log(train_set_batches)

        for batch_idx, (id_data, id_target) in enumerate(id_iterator):
            #sample od ref_data
            try:
                od_data, od_target = next(self.od_iterator)
            except StopIteration:
                #when od iterator runs out, jsut start from beginning
                self.od_iterator = iter(out_distribution_loader)
                od_data, od_target = next(self.od_iterator)

            if (id_data.shape[0] < bs) or (od_data.shape[0] < bs):
                continue

            id_data, id_target = id_data.to(self.device), id_target.to(self.device)
            od_data, od_target = od_data.to(self.device), od_target.to(self.device)

            id_adv_samples = id_trades_train_criterion.inner_max(id_data, id_target)
            od_adv_samples = od_criterion.inner_max(od_data, od_target)

            with amp.autocast(enabled=self.mixed_precision):
                if self.od_trades:
                    clean_out, id_adv_out, od_clean_out, od_adv_out = interleave_forward(self.model,
                                                                           [id_data, id_adv_samples,
                                                                            od_data, od_adv_samples])

                    od_trades_target = F.softmax(od_adv_out, dim=1)
                    od_clean_loss = od_clean_criterion(od_adv_samples, od_adv_out, od_data, od_target)
                    od_trades_loss = od_criterion(od_adv_samples, od_adv_out, od_data, od_trades_target)
                    od_loss = od_clean_loss + self.id_beta * od_trades_loss
                else:
                    clean_out, id_adv_out, od_adv_out = interleave_forward(self.model, [id_data, id_adv_samples, od_adv_samples])
                    od_loss = od_criterion(od_adv_samples, od_adv_out, od_data, od_target)

            id_trades_target = F.softmax(clean_out, dim=1)
            clean_loss = ce_loss_clean(id_data, clean_out, id_data, id_target)
            trades_loss = id_trades_train_criterion(id_adv_samples, id_adv_out, id_data, id_trades_target)
            id_loss = clean_loss + self.id_beta * trades_loss
            loss = 0.5 * (id_loss + self.lam * od_loss)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_logger.log(loss)
            lr_logger.log(self.scheduler.get_last_lr()[0])

            # log
            acc_conf_clean(id_data, clean_out, id_data, id_target)
            acc_conf_id(id_adv_samples, id_adv_out, id_data, id_target)
            distance_id(id_adv_samples, id_adv_out, id_data, id_target)

            confidence_od(od_adv_samples, od_adv_out, od_data, od_target)
            distance_od(od_adv_samples, od_adv_out, od_data, od_target)

            self._update_scheduler(epoch + (batch_idx + 1) / train_set_batches)
            self.output_backend.log_batch_summary(epoch, batch_idx, True, losses=losses, loggers=loggers)

        self._update_scheduler(epoch + 1)
        self.output_backend.end_epoch_write_summary(losses, loggers, epoch, True)



    def _get_TRADESACET_config(self):
        config = {'ACET_obj': self.acet_obj, 'lambda': self.lam, 'TRADES weight': self.id_beta, 'OD Trades': self.od_trades}
        return config

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        tradesacet_config = self._get_TRADESACET_config()
        configs = {}
        configs['Base'] = base_config
        configs['TRADESACET'] = tradesacet_config
        configs['ID Attack'] = self.id_attack_config
        configs['OD Attack'] = self.od_attack_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs

