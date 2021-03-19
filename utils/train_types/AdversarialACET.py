import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils.distances as d

from .ACET_training import ACETObjective, ACETTargetedObjective
from .Adversarial_training import AdversarialTraining, AdversarialLoss
from .in_out_distribution_training import InOutDistributionTraining
from .helpers import get_adversarial_attack, create_attack_config, get_distance
from .train_loss import CrossEntropyProxy


######################################################
class AdversarialACET(InOutDistributionTraining):
    def __init__(self, model, id_attack_config, od_attack_config, optimizer_config, epochs, device, num_classes,
                 train_clean=True,
                 attack_loss='LogitsDiff', lr_scheduler_config=None, model_config=None,
                 target_confidences=False,
                 attack_obj='log_conf', train_obj='log_conf', lam=1., test_epochs=1, verbose=100,
                 saved_model_dir='SavedModels', saved_log_dir='Logs'):

        id_distance = get_distance(id_attack_config['norm'])
        od_distance = get_distance(od_attack_config['norm'])

        super().__init__('AdvACET', model, id_distance, od_distance, optimizer_config, epochs, device, num_classes,
                         train_clean=train_clean,
                         lr_scheduler_config=lr_scheduler_config, model_config=model_config,
                         lam=lam, test_epochs=test_epochs,
                         verbose=verbose, saved_model_dir=saved_model_dir, saved_log_dir=saved_log_dir)

        # Adversarial specific
        self.id_attack_config = id_attack_config
        self.attack_loss = attack_loss

        # ACET specifics
        self.target_confidences = target_confidences
        self.od_attack_config = od_attack_config
        self.od_attack_obj = attack_obj
        self.od_train_obj = train_obj

    def _get_train_type_config(self, loader_config=None):
        base_config = self._get_base_config()
        ACET_config = self._get_ACET_config()
        adv_config = self._get_adversarial_training_config()

        configs = {}
        configs['Base'] = base_config
        configs['Adversarial Training'] = adv_config
        configs['ID Attack'] = self.id_attack_config
        configs['ACET'] = ACET_config
        configs['OD Attack'] = self.od_attack_config
        configs['Optimizer'] = self.optimizer_config
        configs['Scheduler'] = self.lr_scheduler_config

        configs['Data Loader'] = loader_config
        configs['MSDA'] = self.msda_config
        configs['Model'] = self.model_config

        return configs

    @staticmethod
    def create_id_attack_config(eps, steps, stepsize, norm, momentum=0.9, pgd='pgd', normalize_gradient=False,
                                noise=None):
        return create_attack_config(eps, steps, stepsize, norm, momentum=momentum, pgd=pgd,
                                    normalize_gradient=normalize_gradient, noise=noise)

    def _get_id_criterion(self, epoch):
        id_train_criterion = AdversarialLoss(self.model, epoch, self.id_attack_config, self.classes,
                                             inner_objective=self.attack_loss,
                                             log_stats=True, number_of_batches=None, name_prefix='ID')
        return id_train_criterion

    def _get_adversarial_training_config(self):
        return AdversarialTraining._get_adversarial_training_config(self)

    @staticmethod
    def create_od_attack_config(eps, steps, stepsize, norm, momentum=0.9, pgd='pgd', normalize_gradient=False,
                                noise=None):
        return create_attack_config(eps, steps, stepsize, norm, momentum=momentum, pgd=pgd,
                                    normalize_gradient=normalize_gradient, noise=noise)

    def _get_od_criterion(self, epoch):
        if self.target_confidences:
            train_criterion = ACETTargetedObjective(self.model, epoch, self.od_attack_config, self.od_train_obj,
                                                    self.od_attack_obj, self.classes,
                                                    log_stats=True, name_prefix='OD')
        else:
            train_criterion = ACETObjective(self.model, epoch, self.od_attack_config, self.od_train_obj,
                                            self.od_attack_obj, self.classes,
                                            log_stats=True, name_prefix='OD')
        return train_criterion

    def _get_od_attack(self, epoch, att_criterion):
        return get_adversarial_attack(self.od_attack_config, self.model, att_criterion, num_classes=self.classes,
                                      epoch=epoch)

    def _get_ACET_config(self):
        ACET_config = {'targeted confidences': self.target_confidences, 'train_obj': self.od_train_obj,
                       'attack_obj': self.od_attack_obj, 'lambda': self.lam}
        return ACET_config
