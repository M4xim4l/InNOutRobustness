import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .train_type import TrainType
from .train_loss import LoggingLoss, TrainLoss, CrossEntropyProxy, AccuracyConfidenceLogger, DistanceLogger,\
    SingleValueLogger, KLDivergenceProxy, NegativeWrapper, MinMaxLoss
from utils.distances import LPDistance
from .helpers import interleave_forward, get_adversarial_attack, create_attack_config, get_distance
import torch.cuda.amp as amp

class TRADESLoss(MinMaxLoss):
    def __init__(self,  model, epoch, attack_config, num_classes, log_stats=False, name_prefix=None):
        super().__init__('TRADES', 'logits', log_stats=log_stats, name_prefix=name_prefix)
        self.model = model
        self.epoch = epoch
        self.attack_config = attack_config

        self.div = KLDivergenceProxy(log_stats=False)
        self.adv_attack = get_adversarial_attack(self.attack_config, self.model, 'kl', num_classes, epoch=self.epoch)

    def inner_max(self, data, target):
        is_train = self.model.training
        #attack is run in test mode so target distribution should also be estimated in test not train
        self.model.eval()
        target_distribution = F.softmax(self.model(data), dim=1).detach()
        x_adv = self.adv_attack(data, target_distribution)

        if is_train:
            self.model.train()
        else:
            self.model.eval()

        return x_adv.detach()

    #model out will be model out at adversarial samples
    #y will be the softmax distribution at original datapoint
    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        prep_out = self._prepare_input(model_out)
        loss_expanded = self.div(data, prep_out, orig_data, y, reduction='none')
        self._log_stats(loss_expanded)
        return TrainLoss.reduce(loss_expanded, reduction)

#Base class for train types that use custom losses/attacks on the in distribution such as adversarial training
class TRADESTraining(TrainType):
    def __init__(self, model, id_attack_config, optimizer_config, epochs, device, num_classes, trades_weight=1.,
                 lr_scheduler_config=None, model_config=None, test_epochs=1, verbose=100,
                 saved_model_dir= 'SavedModels', saved_log_dir= 'Logs'):
        super().__init__('TRADES', model, optimizer_config, epochs, device, num_classes,
                         lr_scheduler_config=lr_scheduler_config, model_config=model_config, test_epochs=test_epochs,
                         verbose=verbose, saved_model_dir= saved_model_dir, saved_log_dir=saved_log_dir)

        self.id_attack_config = id_attack_config
        self.beta = trades_weight

        distance = get_distance(id_attack_config['norm'])
        self.id_distance = distance

        self.best_id_accuracy = 0.0

    @staticmethod
    def create_id_attack_config(eps, steps, stepsize, norm, momentum=0.9, pgd='pgd', normalize_gradient=True, noise=None):
        return create_attack_config(eps, steps, stepsize, norm, momentum=momentum, pgd=pgd, normalize_gradient=normalize_gradient, noise=noise)

    def _get_TRADES_regularizer(self, epoch):
        trades_reg = TRADESLoss(self.model, epoch, self.id_attack_config, self.classes, log_stats=True)
        return trades_reg

    def test(self, test_loaders, epoch, test_avg_model=False):
        self.model.eval()

        test_loader = test_loaders['test_loader']
        test_set_batches = len(test_loader)
        ce_loss_clean = self._get_clean_criterion(log_stats=True, name_prefix='Clean')
        trades_reg = self._get_TRADES_regularizer(0)
        losses = [ce_loss_clean, trades_reg]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        acc_conf_adv = AccuracyConfidenceLogger(name_prefix='ID')
        distance_adv = DistanceLogger(self.id_distance, name_prefix='ID')
        loggers = [acc_conf_clean, acc_conf_adv, distance_adv]

        self.output_backend.start_epoch_log(test_set_batches)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                adv_samples = trades_reg.inner_max(data, target)

                clean_out, adv_out = interleave_forward(self.model, [data, adv_samples])
                trades_target = F.softmax(clean_out, dim=1)

                loss0 = ce_loss_clean(data, clean_out, data, target)
                loss1 = trades_reg(adv_samples, adv_out, data, trades_target)

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
        if log_epoch is None:
            log_epoch = epoch

        self.model.train()
        train_loader = train_loaders['train_loader']

        train_set_batches = self._get_dataloader_length(train_loader)

        ce_loss_clean = self._get_clean_criterion(log_stats=True, name_prefix='Clean')

        trades_reg = self._get_TRADES_regularizer(epoch)
        losses = [ce_loss_clean, trades_reg]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        acc_conf_adv = AccuracyConfidenceLogger(name_prefix='ID')
        distance_adv = DistanceLogger(self.id_distance, name_prefix='ID')
        total_loss_logger = SingleValueLogger('Loss')
        lr_logger = SingleValueLogger('LR')

        loggers = [total_loss_logger, acc_conf_clean, acc_conf_adv, distance_adv, lr_logger]

        id_iterator = iter(train_loader)

        self.output_backend.start_epoch_log(train_set_batches)
        for batch_idx, (id_data, id_target) in enumerate(id_iterator):
            # sample clean ref_data
            id_data, id_target = id_data.to(self.device), id_target.to(self.device)
            id_adv_samples = trades_reg.inner_max(id_data, id_target)

            with amp.autocast(enabled=self.mixed_precision):
                clean_out, adv_out = interleave_forward(self.model, [id_data, id_adv_samples])
                trades_target = F.softmax(clean_out, dim=1)
                loss0 = ce_loss_clean(id_data, clean_out, id_data, id_target)
                loss1 = trades_reg(id_adv_samples, adv_out, id_data, trades_target)

                loss = loss0 + self.beta * loss1

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_logger.log(loss)
            lr_logger.log(self.scheduler.get_last_lr()[0])

            #log
            acc_conf_clean(id_data, clean_out, id_data, id_target)
            acc_conf_adv(id_adv_samples, adv_out, id_data, id_target)
            distance_adv(id_adv_samples, adv_out, id_data, id_target)

            self._update_scheduler(epoch + (batch_idx + 1) / train_set_batches)
            self.output_backend.log_batch_summary(log_epoch, batch_idx, True, losses=losses, loggers=loggers)

        self.output_backend.end_epoch_write_summary(losses, loggers, log_epoch, True)

    def _get_TRADES_config(self):
        return {'beta': self.beta}

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        adv_config = self._get_TRADES_config()

        configs = {}
        configs['Base'] = base_config
        configs['TRADES'] = adv_config
        configs['ID Attack'] = self.id_attack_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs
