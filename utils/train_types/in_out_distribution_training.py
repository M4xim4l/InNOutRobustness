import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.distances as d
import math

from .out_distribution_training import OutDistributionTraining
from .helpers import interleave_forward
from .train_loss import CrossEntropyProxy, AccuracyConfidenceLogger, DistanceLogger, ConfidenceLogger, SingleValueLogger, SingleValueHistogramLogger
import torch.cuda.amp as amp

######################################################

class InOutDistributionTraining(OutDistributionTraining):
    def __init__(self, name, model, id_distance, od_distance, optimizer_config, epochs, device, num_classes,
                 train_clean=True, lam=1, lr_scheduler_config=None, model_config=None, test_epochs=1, verbose=100,
                 saved_model_dir='SavedModels', saved_log_dir='Logs'):

        super().__init__(name, model, od_distance, optimizer_config, epochs, device, num_classes,
                         lr_scheduler_config=lr_scheduler_config, lam=lam, model_config=model_config,
                         test_epochs=test_epochs,
                         verbose=verbose, saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)

        #ID attributes
        self.train_clean = train_clean
        self.id_distance = id_distance
        self.best_id_accuracy = 0.0

    @staticmethod
    def create_id_attack_config(eps, steps, stepsize, norm, pgd='pgd', normalize_gradient=False, noise=None):
        raise NotImplementedError()

    def _get_id_criterion(self, epoch):
        raise NotImplementedError()

    def _get_id_accuracy_conf_logger(self, name_prefix):
        return AccuracyConfidenceLogger(name_prefix=name_prefix)

    def test(self, test_loaders, epoch, test_avg_model=False):
        if test_avg_model:
            model = self.swa_model
        else:
            model = self.model

        model.eval()

        test_loader = test_loaders['test_loader']
        test_set_batches = len(test_loader)
        ce_loss_clean = self._get_clean_criterion(log_stats=True, name_prefix='Clean')
        id_train_criterion = self._get_id_criterion(epoch)
        losses = [ce_loss_clean, id_train_criterion]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        acc_conf_adv = self._get_id_accuracy_conf_logger(name_prefix='ID')
        distance_adv = DistanceLogger(self.id_distance, name_prefix='ID')
        loggers = [acc_conf_clean, acc_conf_adv, distance_adv]

        self.output_backend.start_epoch_log(test_set_batches)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                adv_samples = id_train_criterion.inner_max(data, target)

                clean_out, adv_out = interleave_forward(model, [data, adv_samples])
                loss0 = ce_loss_clean(data, clean_out, data, target)
                loss1 = id_train_criterion(adv_samples, adv_out, data, target)

                acc_conf_clean(data, clean_out, data, target)
                acc_conf_adv(adv_samples, adv_out, data, target)
                distance_adv(adv_samples, adv_out, data, target)
                self.output_backend.log_batch_summary(epoch, batch_idx, False, losses=losses, loggers=loggers)

        self.output_backend.end_epoch_write_summary(losses, loggers, epoch, False)
        id_acc = acc_conf_adv.get_accuracy()
        if id_acc > self.best_id_accuracy:
            new_best = True
            self.best_id_accuracy = id_acc
        else:
            new_best = False

        return new_best


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

        clean_loss = self._get_clean_criterion()
        id_train_criterion = self._get_id_criterion(epoch)
        od_train_criterion = self._get_od_criterion(epoch)
        losses = [clean_loss, id_train_criterion, od_train_criterion]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        acc_conf_id = self._get_id_accuracy_conf_logger(name_prefix='ID')
        distance_id = DistanceLogger(self.id_distance, name_prefix='ID')

        confidence_od = self._get_od_conf_logger(name_prefix='OD')
        distance_od = DistanceLogger(self.od_distance, name_prefix='OD')
        total_loss_logger = SingleValueLogger('Loss')
        lr_logger = SingleValueLogger('LR')
        loggers = [total_loss_logger, acc_conf_id, distance_id, acc_conf_clean, confidence_od, distance_od, lr_logger]

        self.output_backend.start_epoch_log(train_set_batches)

        for batch_idx, (id_data, id_target) in enumerate(id_iterator):
            #sample clean ref_data
            if self.train_clean:
                try:
                    clean_data, clean_target = next(id_iterator)
                    clean_data, clean_target = clean_data.to(self.device), clean_target.to(self.device)
                except StopIteration:
                    #when id iterator runs out, end epoch
                    break

            #sample od ref_data
            try:
                od_data, od_target = next(self.od_iterator)
            except StopIteration:
                #when od iterator runs out, jsut start from beginning
                self.od_iterator = iter(out_distribution_loader)
                od_data, od_target = next(self.od_iterator)

            if (id_data.shape[0] < bs) or (od_data.shape[0] < bs) or (self.train_clean and clean_data.shape[0] < bs):
                continue

            id_data, id_target = id_data.to(self.device), id_target.to(self.device)
            od_data, od_target = od_data.to(self.device), od_target.to(self.device)

            #id_attack
            id_adv_samples = id_train_criterion.inner_max(id_data, id_target)

            #od attack
            od_adv_samples = od_train_criterion.inner_max(od_data, od_target)

            with amp.autocast(enabled=self.mixed_precision):
                if self.train_clean:
                    clean_out, id_adv_out, od_adv_out = interleave_forward(self.model, [clean_data, id_adv_samples, od_adv_samples])
                    loss0 = clean_loss(clean_data, clean_out, clean_data, clean_target)
                else:
                    id_adv_out, od_adv_out = interleave_forward(self.model, [id_adv_samples, od_adv_samples])
                    loss0 = torch.tensor(0.0, device=self.device)

                loss1 = id_train_criterion(id_adv_samples, id_adv_out, id_data, id_target)
                loss2 = od_train_criterion(od_adv_samples, od_adv_out, od_data, od_target)

                if self.train_clean:
                    loss = (1/3) * (loss0 + loss1 + self.lam * loss2)
                else:
                    loss = (1/2)* (loss1 + self.lam * loss2)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.optimizer.step()

            total_loss_logger.log(loss)
            lr_logger.log(self.scheduler.get_last_lr()[0])

            # log
            if self.train_clean:
                acc_conf_clean(clean_data, clean_out, clean_data, clean_target)

            acc_conf_id(id_adv_samples, id_adv_out, id_data, id_target)
            distance_id(id_adv_samples, id_adv_out, id_data, id_target)

            confidence_od(od_adv_samples, od_adv_out, od_data, od_target)
            distance_od(od_adv_samples, od_adv_out, od_data, od_target)

            self._update_scheduler(epoch + (batch_idx + 1) / train_set_batches)
            self.output_backend.log_batch_summary(epoch, batch_idx, True, losses=losses, loggers=loggers)

        self._update_scheduler(epoch + 1)
        self.output_backend.end_epoch_write_summary(losses, loggers, epoch, True)

    def _update_swa_batch_norm(self, train_loaders):
        train_loader = train_loaders['train_loader']
        out_distribution_loader = train_loaders['out_distribution_loader']

        id_iterator = iter(train_loader)

        clean_loss = self._get_clean_criterion()
        id_train_criterion = self._get_id_criterion(0)
        od_train_criterion = self._get_od_criterion(0)
        bs = self._get_loader_batchsize(train_loader)
        od_bs = self._get_loader_batchsize(out_distribution_loader)

        self.swa_model.train()

        with torch.no_grad():
            for batch_idx, (id_data, id_target) in enumerate(id_iterator):
                if self.train_clean:
                    try:
                        clean_data, clean_target = next(id_iterator)
                        clean_data, clean_target = clean_data.to(self.device), clean_target.to(self.device)
                    except StopIteration:
                        #when id iterator runs out, end epoch
                        break

                #sample od ref_data
                try:
                    od_data, od_target = next(self.od_iterator)
                except StopIteration:
                    #when od iterator runs out, jsut start from beginning
                    self.od_iterator = iter(out_distribution_loader)
                    od_data, od_target = next(self.od_iterator)

                if (id_data.shape[0] < bs) or (od_data.shape[0] < bs) or (self.train_clean and clean_data.shape[0] < bs):
                    continue

                id_data, id_target = id_data.to(self.device), id_target.to(self.device)
                od_data, od_target = od_data.to(self.device), od_target.to(self.device)

                #id_attack
                id_adv_samples = id_train_criterion.inner_max(id_data, id_target)

                #od attack
                od_adv_samples = od_train_criterion.inner_max(od_data, od_target)

                if self.train_clean:
                    clean_out, id_adv_out, od_adv_out = interleave_forward(self.model, [clean_data, id_adv_samples, od_adv_samples])
                else:
                    id_adv_out, od_adv_out = interleave_forward(self.model, [id_adv_samples, od_adv_samples])
