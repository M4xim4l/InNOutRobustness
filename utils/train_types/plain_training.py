import torch
from .train_type import TrainType
from .train_loss import CrossEntropyProxy, AccuracyConfidenceLogger, SingleValueLogger
import torch.cuda.amp as amp

class PlainTraining(TrainType):
    def __init__(self, model, optimizer_config, epochs, device, num_classes, clean_criterion='ce',
                 lr_scheduler_config=None, msda_config=None, model_config=None, test_epochs=1, verbose=100, saved_model_dir='SavedModels',
                 saved_log_dir='Logs'):
        super().__init__('plain', model, optimizer_config, epochs, device, num_classes,
                         lr_scheduler_config=lr_scheduler_config, msda_config=msda_config, model_config=model_config,
                         test_epochs=test_epochs, clean_criterion=clean_criterion,
                         verbose=verbose, saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)


    def _inner_train(self, train_loaders, epoch, log_epoch=None):
        if log_epoch is None:
            log_epoch = epoch

        train_loader = train_loaders['train_loader']
        self.model.train()
        train_set_batches = self._get_dataloader_length(train_loader)

        clean_loss = self._get_clean_criterion(log_stats=True, name_prefix='Clean')
        clean_loss, msda = self._get_msda(clean_loss, log_stats=True, name_prefix='Clean')
        losses = [clean_loss]
        acc_conf = self._get_clean_accuracy_conf_logger(name_prefix='Clean')
        lr_logger = SingleValueLogger('LR')
        loggers = [acc_conf, lr_logger]

        self.output_backend.start_epoch_log(train_set_batches)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = msda(data)
            data, target = data.to(self.device), target.to(self.device)

            with amp.autocast(enabled=self.mixed_precision):
                output = self.model(data)
                loss = clean_loss(data, output, data, target)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc_conf(data, output, data, target)
            lr_logger.log(self.scheduler.get_last_lr()[0])

            #ema
            if self.ema:
                self._update_avg_model()

            self._update_scheduler(epoch + (batch_idx + 1) / train_set_batches)
            self.output_backend.log_batch_summary(log_epoch, batch_idx, True, losses=losses, loggers=loggers)

        self.output_backend.end_epoch_write_summary(losses, loggers, log_epoch, True)

    def _update_avg_model_batch_norm(self, train_loaders):
        train_loader = train_loaders['train_loader']
        self.avg_model.train()
        clean_loss = self._get_clean_criterion(log_stats=False)
        clean_loss, msda = self._get_msda(clean_loss, log_stats=False)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                data = msda(data)
                data, target = data.to(self.device), target.to(self.device)
                output = self.avg_model(data)

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        configs = {}
        configs['Base'] = base_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs