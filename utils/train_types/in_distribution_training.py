import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .train_type import TrainType
from .train_loss import TrainLoss, CrossEntropyProxy, AccuracyConfidenceLogger, DistanceLogger, SingleValueLogger

from utils.adversarial_attacks import *
from utils.distances import LPDistance
from .helpers import interleave_forward
import torch.cuda.amp as amp

#Base class for train types that use custom losses/attacks on the in distribution such as adversarial training
class InDistributionTraining(TrainType):
    def __init__(self, name, model, id_distance, optimizer_config, epochs, device, num_classes,
                 clean_criterion='ce', train_clean=True, lr_scheduler_config=None, model_config=None,
                 test_epochs=1, verbose=100, saved_model_dir='SavedModels', saved_log_dir='Logs'):
        super().__init__(name, model, optimizer_config, epochs, device, num_classes,
                         clean_criterion=clean_criterion,
                         lr_scheduler_config=lr_scheduler_config, model_config=model_config, test_epochs=test_epochs,
                         verbose=verbose, saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)

        self.train_clean = train_clean
        self.id_distance = id_distance
        self.best_id_accuracy = 0.0

    def _get_id_criterion(self, epoch):
        raise NotImplementedError()

    def _get_id_accuracy_conf_logger(self, name_prefix):
        return AccuracyConfidenceLogger(name_prefix=name_prefix)

    def test(self, test_loaders, epoch, test_avg_model=False):
        if test_avg_model:
            raise NotImplementedError()
            #make sure that eval attack is performed on avg density_model
            model = self.swa_model
        else:
            model = self.model

        model.eval()

        new_best = False

        if 'test_loader' in test_loaders:
            test_loader = test_loaders['test_loader']
            id_acc = self._inner_test(model, test_loader, epoch, prefix='Clean', id_prefix='ID')
            if id_acc > self.best_id_accuracy:
                new_best = True
                self.best_id_accuracy = id_acc

        if 'extra_test_loaders' in test_loaders:
            for i, test_loader in enumerate(test_loaders['extra_test_loaders']):
                prefix = f'CleanExtra{i}'
                id_prefix = f'IDExtra{i}'
                self._inner_test(model, test_loader, epoch, prefix=prefix, id_prefix=id_prefix)


        return new_best

    def _inner_test(self, model, test_loader, epoch, prefix='Clean', id_prefix='ID', *args, **kwargs):
        test_set_batches = len(test_loader)
        clean_loss = self._get_clean_criterion(log_stats=True, name_prefix=prefix)

        id_train_criterion = self._get_id_criterion(0) #set 0 as epoch so it uses same attack steps every time
        losses = [clean_loss, id_train_criterion]

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
                loss0 = clean_loss(data, clean_out, data, target)
                loss1 = id_train_criterion(adv_samples, adv_out, data, target)

                acc_conf_clean(data, clean_out, data, target)
                acc_conf_adv(adv_samples, adv_out, data, target)
                distance_adv(adv_samples, adv_out, data, target)
                self.output_backend.log_batch_summary(epoch, batch_idx, False, losses=losses, loggers=loggers)

        self.output_backend.end_epoch_write_summary(losses, loggers, epoch, False)
        id_acc = acc_conf_adv.get_accuracy()
        return id_acc

    def _inner_train(self, train_loaders, epoch, log_epoch=None):
        if log_epoch is None:
            log_epoch = epoch

        self.model.train()
        train_loader = train_loaders['train_loader']

        train_set_batches = self._get_dataloader_length(train_loader)
        bs = self._get_loader_batchsize(train_loader)

        clean_loss = self._get_clean_criterion(log_stats=True, name_prefix='Clean')
        id_train_criterion = self._get_id_criterion(epoch)
        losses = [clean_loss, id_train_criterion]

        acc_conf_clean = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        acc_conf_adv = self._get_id_accuracy_conf_logger(name_prefix='ID')
        distance_adv = DistanceLogger(self.id_distance, name_prefix='ID')
        total_loss_logger = SingleValueLogger('Loss')
        lr_logger = SingleValueLogger('LR')
        loggers = [total_loss_logger, acc_conf_clean, acc_conf_adv, distance_adv, lr_logger]

        id_iterator = iter(train_loader)

        self.output_backend.start_epoch_log(train_set_batches)

        for batch_idx, (id_data, id_target) in enumerate(id_iterator):
            # sample clean ref_data

            if self.train_clean:
                try:
                    clean_data, clean_target = next(id_iterator)
                    clean_data, clean_target = clean_data.to(self.device), clean_target.to(self.device)
                except StopIteration:
                    break

            if id_data.shape[0] < bs or (self.train_clean and clean_data.shape[0] < bs):
                continue

            id_data, id_target = id_data.to(self.device), id_target.to(self.device)
            id_adv_samples = id_train_criterion.inner_max(id_data, id_target)
            with amp.autocast(enabled=self.mixed_precision):

                if self.train_clean:
                    clean_out, adv_out = interleave_forward(self.model, [clean_data, id_adv_samples])
                    loss0 = clean_loss(clean_data, clean_out, clean_data, clean_target)
                    loss1 = id_train_criterion(id_adv_samples, adv_out, id_data, id_target)
                    loss = 0.5 * (loss0 + loss1)
                else:
                    adv_out = self.model(id_adv_samples)
                    loss = id_train_criterion(id_adv_samples, adv_out, id_data, id_target)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss_logger.log(loss)
            lr_logger.log(self.scheduler.get_last_lr()[0])

            #log
            if self.train_clean:
                acc_conf_clean(clean_data, clean_out, clean_data, clean_target)

            acc_conf_adv(id_adv_samples, adv_out, id_data, id_target)
            distance_adv(id_adv_samples, adv_out, id_data, id_target)

            self._update_scheduler(epoch + (batch_idx + 1) / train_set_batches)
            self.output_backend.log_batch_summary(log_epoch, batch_idx, True, losses=losses, loggers=loggers)

        self._update_scheduler(epoch + 1)
        self.output_backend.end_epoch_write_summary(losses, loggers, log_epoch, True)
