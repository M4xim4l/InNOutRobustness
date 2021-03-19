import matplotlib as mpl

mpl.use('Agg')

import os

import torch
import torch.nn as nn
from utils.model_normalization import Cifar100Wrapper
import utils.datasets as dl
import utils.run_file_helpers as rh
import utils.models.model_factory_32 as factory

import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser.add_argument('--net', type=str, default='ResNet18', help='Resnet18, 34 or 50, WideResNet28')
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 or semi-cifar10')
parser.add_argument('--od_dataset', type=str, default='tinyImages',
                    help=('tinyImages or cifar10'))
parser.add_argument('--exclude_cifar', dest='exclude_cifar', type=lambda x: bool(strtobool(x)),
                    default=True, help='whether to exclude cifar10 from tiny images')

rh.parser_add_commons(parser)
rh.parser_add_adversarial_commons(parser)
rh.parser_add_adversarial_norms(parser, 'cifar100')

hps = parser.parse_args()

if len(hps.gpu) == 0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
elif len(hps.gpu) == 1:
    device = torch.device('cuda:' + str(hps.gpu[0]))
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))

#FOLDERS
num_classes = 100
img_size = 32
model_root_dir = 'Cifar100Models'
logs_root_dir = 'Cifar100Logs'
model, model_name = factory.build_model(hps.net, num_classes)
model_dir = os.path.join(model_root_dir, model_name)
log_dir = os.path.join(logs_root_dir, model_name)
start_epoch, optim_state_dict = rh.load_model_checkpoint(model, model_dir, device, hps)
model = Cifar100Wrapper(model).to(device)
if len(hps.gpu) > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

msda_config = rh.create_msda_config(hps)

#load dataset
od_bs = int(hps.od_bs_factor * hps.bs)

if hps.train_type in ['CEDA', 'ACET', 'AdvACET','ADVACET']:
    id_config = {}
    od_config = {}
    loader_config = {'ID config': id_config, 'OD config': od_config}

    train_loader = dl.get_CIFAR100(train=True, batch_size=hps.bs, augm_type=hps.augm,
                                   config_dict=id_config)

    if hps.od_dataset == 'tinyimages':
        tiny_train = dl.get_80MTinyImages(batch_size=od_bs, augm_type=hps.augm, exclude_cifar=hps.exclude_cifar,
                                          exclude_cifar10_1=False,
                                          config_dict=od_config)
    elif hps.od_dataset == 'cifar10':
        tiny_train = dl.get_CIFAR10(train=True, batch_size=od_bs, shuffle=True, augm_type=hps.augm,
                                    config_dict=od_config)
    else:
        raise ValueError('OD Dataset not supported')
else:
    id_config = {}
    loader_config = {'ID config': id_config}
    train_loader = dl.get_CIFAR100(train=True, batch_size=hps.bs, augm_type=hps.augm,
                                   config_dict=id_config)

test_loader = dl.get_CIFAR100(train=False, batch_size=hps.bs, augm_type='none')

scheduler_config, optimizer_config = rh.create_optim_scheduler_swa_configs(hps)

trainer = rh.create_trainer(hps, model, optimizer_config, scheduler_config, device, num_classes,
                            model_dir, log_dir, msda_config)

##DEBUG:
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

# run training
if trainer.requires_out_distribution():
    train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader,
                                                              out_distribution_loader=tiny_train)
    trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                  optim_state_dict=optim_state_dict)
else:
    train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader)
    trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                  optim_state_dict=optim_state_dict)
