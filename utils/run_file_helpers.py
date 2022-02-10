import torch
from distutils.util import strtobool
import utils.train_types.schedulers as schedulers
import utils.train_types.optimizers as optimizers
import utils.train_types.msda as msda
import utils.train_types as tt
import math
import numpy as np

def parser_add_commons(parser):
    parser.add_argument('--gpu', '--list', nargs='+', default=[0],
                        help='GPU indices, if more than 1 parallel modules will be called')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for training')
    parser.add_argument('--bs', type=int, default=128, help='Training batch_size')
    parser.add_argument('--decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs')
    parser.add_argument('--test_epochs', type=int, default=1, help='Test frequency')
    parser.add_argument('--train_type', type=str, default='plain', help='Train type')
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizer')
    parser.add_argument('--continue', dest='continue_trained', nargs=3, type=str,
                        default=None, help='Filename Load_epoch Start_epoch of model to continue')
    parser.add_argument('--augm', type=str, default='default',
                        help=('Augmentation type'))
    parser.add_argument('--warmup_epochs', type=int,
                        default='0', help='Warmup length')
    parser.add_argument('--schedule', type=str,
                        default='step_lr', help='LR scheduler')
    parser.add_argument('--nesterov', dest='nesterov', type=lambda x: bool(strtobool(x)),
                        default=True, help='Nesterov SGD')
    parser.add_argument('--msda', type=str,
                        default='none', help='MSDA: None, mixup or fmix')
    parser.add_argument('--msda_alpha', type=float,
                        default=1.0, help='MSDA Alpha')
    parser.add_argument('--mixed_precision', type=lambda x: bool(strtobool(x)),
                        default=False, help='Mixed precision training')
    parser.add_argument('--ema', type=lambda x: bool(strtobool(x)),
                        default=False, help='Exponential moving average')
    parser.add_argument('--ema_decay', type=float,
                        default=0.999,
                        help='EMA decay')
    parser.add_argument('--sam_rho', type=float,
                        default=0.05,
                        help='SAM rho')
    parser.add_argument('--sam_adaptive', type=lambda x: bool(strtobool(x)),
                        default=False,
                        help='SAM adaptive toggle')

    parser.add_argument('--swa', type=str,
                        default=None, help='SWA cosine or const')
    parser.add_argument('--swa_epochs', type=int,
                        default=500, help='SWA epochs after regular training')
    parser.add_argument('--swa_cycle_length', type=int,
                        default=200,
                        help='In Cosine mode, SWA repeats epochs [SWA_END - SWA_CYCLE_LENGTH, SWA_END] for a total number of SWA_epochs')
    parser.add_argument('--swa_virtual_schedule_length', type=int,
                        default=1800,
                        help='In Cosine mode, SWA repeats epochs [SWA_END - SWA_CYCLE_LENGTH, SWA_END] for a total number of SWA_epochs')
    parser.add_argument('--swa_virtual_schedule_swa_end', type=int,
                        default=1500,
                        help='In Cosine mode, SWA repeats epochs [SWA_END - SWA_CYCLE_LENGTH, SWA_END] for a total number of SWA_epochs')
    parser.add_argument('--swa_lr', type=float,
                        default=0.025,
                        help='SWA LR')
    parser.add_argument('--swa_update_frequency', type=int,
                        default=20, help='SWA epochs after regular training')



def parser_add_adversarial_commons(parser):
    parser.add_argument('--od_weight', type=float, default=1., help='Weight for out-distribution term in ACET (derivates)')
    parser.add_argument('--trades_weight', type=float, default=6., help='Weight for TRADES term in TRADES (derivates)')
    parser.add_argument('--od_trades_weight', type=float, default=6., help='Weight for OD-TRADES term')
    parser.add_argument('--train_clean', dest='train_clean', type=lambda x: bool(strtobool(x)),
                        default=False, help='Train on additional clean data or purely adversarial')
    parser.add_argument('--norm', type=str, default='l2',
                        help=('l2 or linf'))
    parser.add_argument('--od_bs_factor', default=1, type=float, help='OD batch out_size factor')

    #ID PGD for AT/TRADES
    parser.add_argument('--id_steps', type=int, default=20, help='steps in ID attack')
    parser.add_argument('--id_pgd', type=str, default='argmin',
                        help='PGD variation for ID attack: pgd, argmin, monotone')

    #OD PGD for ACET
    parser.add_argument('--od_steps', type=int, default=20, help='steps in OD attack (ACET)')
    parser.add_argument('--od_pgd', type=str, default='argmin',
                        help='PGD variation for OD attack: pgd, argmin, monotone')

    #Objectives
    parser.add_argument('--adv_obj', type=str, default='ce',
                        help=('Objective to optimize in the inner loop of adversarial training'
                              'logitsDiff | crossEntropy'))
    parser.add_argument('--ceda_obj', type=str, default='KL',
                        help=('only for ACET; what objective the adversary has'
                              'conf | log_conf | entropy | KL | bhattacharyya'))

    #Randomized smoothing
    parser.add_argument('--rs_levels', type=int, default='500',
                        help=('Number of randomized smoothing levels'))
    parser.add_argument('--rs_sigma_begin', type=float, default='1.0',
                        help=('Randomized smoothing: start sigma'))
    parser.add_argument('--rs_sigma_end', type=float, default='0.0001',
                        help=('Randomized smoothing: end sigma'))


def parser_add_adversarial_norms(parser, dataset):
    parser.add_argument('--eps', type=float, default=None, help='Epsilon')
    parser.add_argument('--stepsize', type=float, default=None, help='PGD stepsize')
    parser.add_argument('--od_eps_factor', type=float, default=1.0, help='Multiplier for ACET epsilon')


def load_model_checkpoint(model, model_dir, device, hps):
    # load old density_model
    if hps.continue_trained is not None:
        load_folder = hps.continue_trained[0]
        load_epoch = hps.continue_trained[1]
        start_epoch = int(int(hps.continue_trained[2]))  # * epoch_subdivs)
        if load_epoch in ['final', 'best', 'best_avg', 'final_avg']:
            state_dict_file = f'{model_dir}/{load_folder}/{load_epoch}.pth'
            optimizer_dict_file = f'{model_dir}/{load_folder}/{load_epoch}_optim.pth'
        else:
            state_dict_file = f'{model_dir}/{load_folder}/checkpoints/{load_epoch}.pth'
            optimizer_dict_file = f'{model_dir}/{load_folder}/checkpoints/{load_epoch}_optim.pth'

        state_dict = torch.load(state_dict_file, map_location=device)

        try:
            optim_state_dict = torch.load(optimizer_dict_file, map_location=device)
        except:
            print('Warning: Could not load Optim State - Restarting optim')
            optim_state_dict = None

        model.load_state_dict(state_dict)

        print(f'Continuing {load_folder} from epoch {load_epoch} - Starting training at epoch {start_epoch}')
    else:
        start_epoch = 0
        optim_state_dict = None

    return start_epoch, optim_state_dict

def create_msda_config(hps):
    if hps.msda == 'none':
        msda_config = None
    elif hps.msda == 'fmix':
        if '_cutout' in hps.augm:
            augm_new = hps.augm.split('_cutout')[0]
            print(f'Warning found augmentation {hps.augm} with cutout - changing to {augm_new}')
            hps.augm = augm_new
        msda_config = msda.create_fmix_config()
    else:
        raise NotImplementedError()

    return msda_config

def create_optim_scheduler_swa_configs(hps):
    # SCHEDULER
    epochs = hps.epochs
    warmup_epochs = hps.warmup_epochs
    decay_rate = 0.1

    if hps.schedule == 'step_lr':
        if epochs == 25:
            decay_epochs = [10, 15, 20]
        if epochs == 50:
            decay_epochs = [25, 40]
        elif epochs == 100:
            decay_epochs = [50, 75, 90]
        elif epochs == 110:
            decay_epochs = [100]
        elif epochs == 120:
            decay_epochs = [50, 85, 105]
        elif epochs == 150:
            decay_epochs = [60, 90, 120]
        elif epochs == 200:
            decay_epochs = [75, 125, 175]
        elif epochs == 220 or epochs == 250 or epochs == 230:
            decay_epochs = [100, 150, 200]
        elif epochs == 300:
            decay_epochs = [80, 160, 240]
        elif epochs == 333:
            decay_epochs = [150, 200, 280]
        elif epochs == 320 or epochs == 350:
            decay_epochs = [150, 225, 300]
        elif epochs == 500:
            decay_epochs = [200, 300, 400]
        elif epochs == 1000:
            decay_epochs = [400, 600, 800]
        elif epochs == 5000:
            decay_epochs = [2000, 3000, 4000]
        else:
            raise NotImplementedError(f'Epochs {epochs} not supported')

        scheduler_config = schedulers.create_piecewise_consant_scheduler_config(epochs, decay_epochs, decay_rate,
                                                                                warmup_length=warmup_epochs)
    elif hps.schedule == 'cosine':
        scheduler_config = schedulers.create_cosine_annealing_scheduler_config(epochs, 0.,
                                                                               warmup_length=warmup_epochs)
    else:
        raise NotImplementedError()

    # optimizer
    if hps.optim.lower() == 'sam':
        optimizer_config = optimizers.create_sam_optimizer_config(hps.lr, momentum=hps.momentum, weight_decay=hps.decay,
                                                                  sam_adaptive=hps.sam_adaptive, sam_rho=hps.sam_rho,
                                                                  nesterov=hps.nesterov,
                                                                  ema=hps.ema, ema_decay=hps.ema_decay)
    else:
        optimizer_config = optimizers.create_optimizer_config(hps.optim, hps.lr, momentum=hps.momentum, weight_decay=hps.decay,
                                                              nesterov=hps.nesterov, mixed_precision=hps.mixed_precision,
                                                              ema=hps.ema, ema_decay=hps.ema_decay)

    if hps.swa == 'const':
        optimizers.config_creators.add_constant_swa_to_optimizer_config(hps.swa_epochs,
                                                                        hps.swa_update_frequency,
                                                                        hps.swa_lr,
                                                                        optimizer_config)
    elif hps.swa == 'cosine':
        optimizers.config_creators.add_cosine_swa_to_optimizer_config(hps.swa_epochs, hps.swa_cycle_length,
                                                                      hps.swa_update_frequency,
                                                                      hps.swa_virtual_schedule_length,
                                                                      hps.swa_virtual_schedule_swa_end,
                                                                      hps.swa_lr,
                                                                      optimizer_config)

    return scheduler_config, optimizer_config


def create_attack_config(hps, dataset):
    if hps.norm.lower() in ['l2', '2']:
        hps.norm = 'l2'
    elif hps.norm.lower() in ['l1', '1']:
        hps.norm = 'l1'
    elif hps.norm.lower() in ['linf', 'inf']:
        hps.norm = 'linf'
    elif hps.norm.lower() in ['l1.5', '1.5']:
        hps.norm = 'l1.5'
    else:
        raise NotImplementedError()

    norm_eps = {}
    norm_stepsizes = {}

    if dataset in ['cifar10', 'cifar100']:
        norm_eps['linf'] = 8 / 255
        norm_eps['l2']  = 0.5
        norm_eps['l1']  = 12
        norm_eps['l1.5']  = 6

        norm_stepsizes['linf'] = 2 / 255
        norm_stepsizes['l2'] = 0.1
        norm_stepsizes['l1'] = 5
        norm_stepsizes['l1.5'] = None
    elif dataset in ['restrictedImagenet', 'imagenet', 'lsun', 'celebA']:
        norm_eps['linf'] = 8 / 255
        norm_eps['l2'] = 3.0
        norm_eps['l1'] = 72
        norm_eps['l1.5'] = 20

        norm_stepsizes['linf'] = 2 / 255
        norm_stepsizes['l2'] = 0.6
        norm_stepsizes['l1'] = 30
        norm_stepsizes['l1.5'] = None
    else:
        raise NotImplementedError()

    if hps.eps is not None:
        eps = hps.eps
    else:
        eps = norm_eps[hps.norm]

    if hps.stepsize is not None:
        stepsize = hps.stepsize
    else:
        stepsize = norm_stepsizes[hps.norm]

    od_eps = eps * hps.od_eps_factor

    id_attack_config = tt.create_attack_config(eps, hps.id_steps, stepsize, hps.norm,
                                               pgd=hps.id_pgd,
                                               normalize_gradient=True)
    od_attack_config = tt.create_attack_config(od_eps, hps.od_steps, stepsize, hps.norm,
                                               pgd=hps.od_pgd,
                                               normalize_gradient=True)

    return id_attack_config, od_attack_config

def create_trainer(hps, model, optimizer_config, scheduler_config, device, num_classes, model_dir, log_dir,
                   msda_config=None, model_config=None, id_attack_config=None, od_attack_config=None):

    if hps.train_type.lower() == 'plain':
        trainer = tt.PlainTraining(model, optimizer_config, hps.epochs, device, num_classes,
                                   lr_scheduler_config=scheduler_config,
                                   msda_config=msda_config, model_config=model_config,
                                   saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    elif hps.train_type.lower() == 'adversarial':
        # https://arxiv.org/pdf/1906.09453.pdf

        trainer = tt.AdversarialTraining(model, id_attack_config, optimizer_config, hps.epochs, device, num_classes,
                                         train_clean=hps.train_clean, attack_loss=hps.adv_obj,
                                         lr_scheduler_config=scheduler_config, model_config=model_config,
                                         saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    elif hps.train_type.lower() in ['trades']:
        trainer = tt.TRADESTraining(model, id_attack_config, optimizer_config, hps.epochs, device, num_classes,
                                    lr_scheduler_config=scheduler_config, model_config=model_config,
                                    trades_weight=hps.trades_weight,
                                    saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    elif hps.train_type.lower() in ['advacet']:
        # https://arxiv.org/pdf/1906.09453.pdf
        trainer = tt.AdversarialACET(model, id_attack_config, od_attack_config, optimizer_config, hps.epochs, device,
                                     num_classes, train_clean=hps.train_clean, attack_loss=hps.adv_obj,
                                     lr_scheduler_config=scheduler_config, model_config=model_config,
                                     train_obj=hps.ceda_obj, attack_obj=hps.ceda_obj, od_weight=hps.od_weight,
                                     saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    elif hps.train_type.lower()  == 'tradesacet':
        trainer = tt.TRADESACETTraining(model, id_attack_config, od_attack_config, optimizer_config, hps.epochs, device,
                                        num_classes, trades_weight=hps.trades_weight,
                                        lr_scheduler_config=scheduler_config, model_config=model_config,
                                        acet_obj=hps.ceda_obj, od_weight=hps.od_weight,
                                        saved_model_dir=model_dir, saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    elif hps.train_type.lower()  == 'tradesceda':
        trainer = tt.TRADESCEDATraining(model, id_attack_config, od_attack_config, optimizer_config, hps.epochs, device,
                                        num_classes, id_trades_weight=hps.trades_weight, od_trades_weight=hps.od_trades_weight,
                                        lr_scheduler_config=scheduler_config,
                                        ceda_obj=hps.ceda_obj, od_weight=hps.od_weight, model_config=model_config,
                                        test_epochs=hps.test_epochs, saved_model_dir=model_dir, saved_log_dir=log_dir)
    elif hps.train_type == 'CEDA':
        trainer = tt.CEDATraining(model, optimizer_config, hps.epochs, device, num_classes, lr_scheduler_config=scheduler_config,
                                  msda_config=msda_config, model_config=model_config,
                                  train_obj=hps.ceda_obj, od_weight=hps.od_weight, saved_model_dir=model_dir,
                                  saved_log_dir=log_dir, test_epochs=hps.test_epochs)
    elif hps.train_type.lower() == 'acet':
        # L2 disance between cifar10 and mnist is about 14 on average
        trainer = tt.ACETTraining(model, od_attack_config, optimizer_config, hps.epochs, device, num_classes,
                                  lr_scheduler_config=scheduler_config, model_config=model_config,
                                  od_weight=hps.od_weight, train_obj=hps.ceda_obj, attack_obj=hps.ceda_obj,
                                  test_epochs=hps.test_epochs, saved_model_dir=model_dir, saved_log_dir=log_dir)
    elif hps.train_type.lower() in ['randomizedsmoothing', 'randomized_smoothing']:
        rs_noise_scales = torch.FloatTensor(np.geomspace(hps.rs_sigma_begin, hps.rs_sigma_end, hps.rs_levels))
        trainer = tt.RandomizedSmoothingTraining(model, optimizer_config, hps.epochs, device, num_classes, rs_noise_scales,
                                                 train_clean=hps.train_clean, lr_scheduler_config=scheduler_config,
                                                 model_config=model_config,
                                                  saved_model_dir=model_dir, saved_log_dir=log_dir,
                                                 test_epochs=hps.test_epochs)
    else:
        raise ValueError('Train type {} is not supported'.format(hps.train_type))

    return trainer

