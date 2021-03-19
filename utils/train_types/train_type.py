import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
import torch.cuda.amp as amp
import copy
import numpy as np
import matplotlib.pyplot as plt
import time


from utils.model_normalization import NormalizationWrapper
from .output_backend import OutputBackend
from .train_loss import CrossEntropyProxy, AccuracyConfidenceLogger, BCELogitsProxy, BCAccuracyConfidenceLogger,\
    KLDivergenceProxy, ConfidenceLogger, KLDivergenceEntropyMinimizationProxy
from .schedulers import create_scheduler, create_cosine_annealing_scheduler_config, create_piecewise_consant_scheduler_config
from .msda.factory import get_msda


class TrainType:
    def __init__(self, type, model, optimizer_config, epochs, device, num_classes, lr_scheduler_config=None,
                 msda_config=None, model_config=None, clean_criterion='crossentropy', test_epochs=5, verbose=100,
                 saved_model_dir='SavedModels', saved_log_dir='Logs'):
        self.type = type
        self.model = model
        self.optimizer_config = optimizer_config
        self.epochs = epochs
        self.device = device
        self.lr_scheduler_config = lr_scheduler_config
        self.msda_config = msda_config
        self.model_config = model_config
        self.clean_criterion = clean_criterion
        self.test_epochs = test_epochs
        self.verbose = verbose

        self.model_dir = saved_model_dir
        self.log_dir= saved_log_dir
        self.best_accuracy = 0.0

        self.classes = num_classes

    def requires_out_distribution(self):
        return False

    def _get_train_type_config(self, loader_config=None):
        raise NotImplementedError()

    def _get_trainset_config(self, train_loader, *args, **kwargs):
        config_dict = {}
        config_dict['batch out_size'] = train_loader.batch_size
        return config_dict

    def _get_base_config(self):
        config_dict = {}
        config_dict['type'] = self.type
        config_dict['epochs'] = self.epochs
        config_dict['clean loss'] = self.clean_criterion
        return config_dict

    def _update_scheduler(self, epoch: float):
        if self.scheduler is not None:
            self.scheduler.step(epoch)

    def _create_optimizer_scheduler(self):
        #OPTIMIZER
        if self.optimizer_config['optimizer_type'] == 'ADAM':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.optimizer_config['lr'], weight_decay=self.optimizer_config['weight_decay'])
        elif self.optimizer_config['optimizer_type'] == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.optimizer_config['lr'], weight_decay=self.optimizer_config['weight_decay'])
        elif self.optimizer_config['optimizer_type'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.optimizer_config['lr'], weight_decay=self.optimizer_config['weight_decay'], momentum=self.optimizer_config['momentum'],
                                        nesterov=self.optimizer_config['nesterov'])
        else:
            raise ValueError('Optimizer not supported {}'.format(self.optimizer_config['optimizer_type']))

        #SWA
        if 'swa_config' in self.optimizer_config:
            self.swa_epochs = self.optimizer_config['swa_config']['epochs']
            self.swa_update_frequency = self.optimizer_config['swa_config']['update_frequency']
            self.swa_virtual_schedule_lr = self.optimizer_config['swa_config']['virtual_schedule_lr']

            if self.optimizer_config['swa_config']['swa_schedule_type'] == 'cosine':
                self.swa_cycle_length = self.optimizer_config['swa_config']['cycle_length']
                swa_virtual_schedule_length = self.optimizer_config['swa_config']['virtual_schedule_length']
                self.swa_virtual_schedule_swa_end = self.optimizer_config['swa_config']['virtual_schedule_swa_end']
                self.swa_scheduler_config = create_cosine_annealing_scheduler_config(swa_virtual_schedule_length, lr_min=0)
            elif self.optimizer_config['swa_config']['swa_schedule_type'] == 'constant':
                self.swa_virtual_schedule_swa_end = self.swa_epochs
                self.swa_cycle_length = self.swa_epochs
                self.swa_scheduler_config = create_piecewise_consant_scheduler_config(self.swa_epochs, [], 1.0)
            else:
                raise NotImplementedError()
        else:
            self.swa_epochs = 0

        #MIXED PRECISION
        if 'mixed_precision' in self.optimizer_config and self.optimizer_config['mixed_precision']:
            print('Using mixed precision training')
            self.scaler = amp.GradScaler(enabled=True)
            self.mixed_precision = True
        else:
            self.scaler = amp.GradScaler(enabled=False)
            self.mixed_precision = False

        #SCHEDULER
        if self.lr_scheduler_config is not None:
            self.scheduler, self.epochs = create_scheduler(self.lr_scheduler_config, self.optimizer)
        else:
            self.scheduler = None

    def _create_swa_optimizer_scheduler(self):
        #set base lr of optimizer to that of our virtual schedule
        for group in self.optimizer.param_groups:
            group['lr'] = self.swa_virtual_schedule_lr

        #create scheduler around old optimizer
        self.scheduler, _ = create_scheduler(self.swa_scheduler_config, self.optimizer)

        #wrap optimizer with swa optimizer
        self.swa_model = AveragedModel(self.model)

    def _get_clean_criterion(self, test=False, log_stats=True, name_prefix=None):
        if self.clean_criterion in ['ce', 'crossentropy']:
            loss = CrossEntropyProxy(log_stats=log_stats, name_prefix=name_prefix)
        elif self.clean_criterion == 'bce':
            loss = BCELogitsProxy(log_stats=log_stats, name_prefix=name_prefix)
        elif self.clean_criterion in ['kl', 'KL']:
            if test:
                loss = CrossEntropyProxy(log_stats=log_stats, name_prefix=name_prefix)
            else:
                loss = KLDivergenceProxy(log_stats=log_stats, name_prefix=name_prefix)
        elif 'klEntropy' in self.clean_criterion:
            #klEntropy_WEIGHT
            if test:
                loss = CrossEntropyProxy(log_stats=log_stats, name_prefix=name_prefix)
            else:
                entropy_weight = float(self.clean_criterion[10:])
                loss = KLDivergenceEntropyMinimizationProxy(entropy_weight=entropy_weight, log_stats=log_stats,
                                                            name_prefix=name_prefix)
        else:
            raise NotImplementedError()

        return loss

    #returns new msda loss and MSDA augmenter
    def _get_msda(self, loss, log_stats=True, name_prefix=None):
        return get_msda(loss, self.msda_config, log_stats=log_stats, name_prefix=name_prefix)

    def _get_clean_accuracy_conf_logger(self, test=False, name_prefix=None):
        if self.clean_criterion in ['ce', 'crossentropy']:
            return AccuracyConfidenceLogger(name_prefix=name_prefix)
        elif self.clean_criterion == 'bce':
            return BCAccuracyConfidenceLogger(self.classes, name_prefix=name_prefix)
        elif self.clean_criterion in ['kl', 'KL'] or 'klEntropy' in self.clean_criterion:
            if test:
                return AccuracyConfidenceLogger(name_prefix=name_prefix)
            else:
                return ConfidenceLogger(name_prefix=name_prefix)
        else:
            raise NotImplementedError()

    def test(self, test_loaders, epoch, test_avg_model=False):
        new_best = False
        if test_avg_model:
            model = self.swa_model
        else:
            model = self.model

        if 'test_loader' in test_loaders:
            test_loader = test_loaders['test_loader']
            test_accuracy = self._inner_test(model, test_loader, epoch, prefix='Clean')
            if test_accuracy > self.best_accuracy:
                new_best = True
                self.best_accuracy = test_accuracy

        if 'extra_test_loaders' in test_loaders:
            for i, test_loader in enumerate(test_loaders['extra_test_loaders']):
                prefix = f'CleanExtra{i}'
                self._inner_test(model, test_loader, epoch, prefix=prefix)

        return new_best

    def _inner_test(self, model, test_loader, epoch, prefix='Clean', *args, **kwargs):
        model.eval()
        clean_loss = self._get_clean_criterion(test=True, log_stats=True, name_prefix=prefix)
        losses = [clean_loss]
        acc_conf = self._get_clean_accuracy_conf_logger(test=True, name_prefix=prefix)
        loggers = [acc_conf]

        test_set_batches = len(test_loader)
        self.output_backend.start_epoch_log(test_set_batches)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)
                clean_loss(data, output, data, target)
                acc_conf(data, output, data, target)
                self.output_backend.log_batch_summary(epoch, batch_idx, False, losses=losses, loggers=loggers)

        self.output_backend.end_epoch_write_summary(losses, loggers, epoch, False)
        test_accuracy = acc_conf.get_accuracy()

        return test_accuracy

    def get_model_state_dict(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()

    def get_swa_model_state_dict(self):
        if isinstance(self.swa_model, torch.nn.DataParallel):
            return self.swa_model.module.state_dict()
        else:
            return self.swa_model.state_dict()


    def get_optimizer_state_dict(self):
        return self.optimizer.state_dict()

    def create_loaders_dict(self, train_loader, test_loader=None, extra_test_loaders=None, *args, **kwargs):
        train_loaders = {
            'train_loader': train_loader
        }

        test_loaders = {}
        if test_loader is not None:
            test_loaders['test_loader'] = test_loader
        if extra_test_loaders is not None:
            test_loaders['extra_test_loaders'] = extra_test_loaders
        return train_loaders, test_loaders

    def _validate_loaders(self, train_loaders, test_loaders):
        if not 'train_loader' in train_loaders:
            raise ValueError('Train cifar_loader not given')

    def _get_loader_batchsize(self, loader):
        if loader.batch_size is not None:
            return loader.batch_size
        else:
            iterator = iter(loader)
            a,b = next(iterator)
            return a.shape[0]

    def _get_dataloader_length(self, loader, *args, **kwargs):
        num_batches = len(loader)
        return num_batches

    def reset_optimizer(self, start_epoch, optim_state_dict):
        if optim_state_dict is not None:
            self.optimizer.load_state_dict(optim_state_dict)

        if start_epoch > 0:
            print(f'Resetting scheduler to epoch: {start_epoch}')
            self._update_scheduler(start_epoch)


    def train(self, train_loaders, test_loaders, loader_config, start_epoch=0, optim_state_dict=None,
              in_parallel=True):
        self._validate_loaders(train_loaders, test_loaders)

        config = self._get_train_type_config(loader_config=loader_config)
        self._create_optimizer_scheduler()
        self.reset_optimizer(start_epoch, optim_state_dict)

        self.output_backend = OutputBackend(self.model_dir, self.log_dir, self.type)
        self.output_backend.save_model_configs(config)

        self.in_parallel = in_parallel

        for epoch in range(start_epoch, self.epochs):
            epoch_start_t = time.time()
            self._inner_train(train_loaders, epoch)

            if (epoch % (5 * self.test_epochs) == 0) or ((epoch / self.epochs >= 0.8) & (epoch % 5 == 0)):
                self.output_backend.save_model_checkpoint(self.get_model_state_dict(), epoch,
                                                          optimizer_state_dict=self.get_optimizer_state_dict())
            if (epoch / self.epochs >= 0.8) | (epoch % self.test_epochs == 0):
                new_best = self.test(test_loaders, epoch, False)
                if new_best:
                    self.output_backend.save_model_checkpoint(self.get_model_state_dict(), 'best',
                                                              optimizer_state_dict=self.get_optimizer_state_dict())

            epoch_t = (time.time() - epoch_start_t) / 60
            self.output_backend.log_epoch_time(epoch_t, epoch, self.epochs)

        self.output_backend.save_model_checkpoint(self.get_model_state_dict(), 'final',
                                                  optimizer_state_dict=self.get_optimizer_state_dict())

        if self.swa_epochs > 0:
            #first train for the desired number of cycle_length then start SWA
            print(f'Starting Stochastic Weight averaging for {self.swa_epochs} epochs')
            self._create_swa_optimizer_scheduler()
            self.scheduler.step(self.swa_virtual_schedule_swa_end - self.swa_cycle_length)

            for swa_epoch in range(0, self.swa_epochs):
                #each cycle repeats the last cycle_length epochs of a virtual schedule with total length
                #swa_virtual_schedule_length
                #so set the epoch in _inner_train accordingly
                scheduler_epoch = self.swa_virtual_schedule_swa_end - (self.swa_cycle_length - (swa_epoch % self.swa_cycle_length))

                #epoch is the total epoch of training, ranging from epochs to epochs + swa_epochs; used for loggig
                epoch = self.epochs + swa_epoch

                epoch_start_t = time.time()
                self._inner_train(train_loaders, scheduler_epoch, log_epoch=epoch)
                epoch_t = (time.time() - epoch_start_t) / 60
                self.output_backend.log_epoch_time(epoch_t, epoch, self.epochs + self.swa_epochs)

                #update the swa density_model every swa_update_frequency epochs
                if ((swa_epoch + 1) % self.swa_update_frequency) == 0:
                    self.swa_model.update_parameters(self.model)

                    if (swa_epoch / self.swa_epochs >= 0.8) | (swa_epoch % self.test_epochs == 0):
                        self._update_swa_batch_norm(train_loaders)
                        new_best = self.test(test_loaders, epoch, True)
                        if new_best:
                            self.output_backend.save_model_checkpoint(self.get_model_state_dict(), 'best_swa')

            self._update_swa_batch_norm(train_loaders)
            self.output_backend.save_model_checkpoint(self.get_model_state_dict(), 'final_swa')

        self.output_backend.close_backend()

    def _inner_train(self, train_loaders, epoch, log_epoch=None):
        raise NotImplementedError()

    def _update_swa_batch_norm(self, train_loaders):
        raise NotImplementedError()