import matplotlib as mpl
mpl.use('Agg')

import os
import torch
import torch.nn as nn

from utils.model_normalization import Cifar10Wrapper
import utils.datasets as dl
import utils.models.model_factory_32 as factory
import utils.run_file_helpers as rh
from distutils.util import strtobool

import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')
parser.add_argument('--net', type=str, default='ResNet18', help='Resnet18, 34 or 50, WideResNet28')
parser.add_argument('--model_params', nargs='+', default=[])
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or semi-cifar10')
parser.add_argument('--od_dataset', type=str, default='tinyImages',
                    help=('tinyImages or cifar100'))
parser.add_argument('--exclude_cifar', dest='exclude_cifar', type=lambda x: bool(strtobool(x)),
                    default=True, help='whether to exclude cifar10 from tiny images')

rh.parser_add_commons(parser)
rh.parser_add_adversarial_commons(parser)
rh.parser_add_adversarial_norms(parser, 'cifar10')

hps = parser.parse_args()
#
device_ids = None
if len(hps.gpu)==0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
elif len(hps.gpu)==1:
    device = torch.device('cuda:' + str(hps.gpu[0]))
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))

#Load model
model_root_dir = 'Cifar10Models'
logs_root_dir = 'Cifar10Logs'
num_classes = 10

model, model_name, model_config, img_size = factory.build_model(hps.net, num_classes, model_params=hps.model_params)
model_dir = os.path.join(model_root_dir, model_name)
log_dir = os.path.join(logs_root_dir, model_name)

start_epoch, optim_state_dict = rh.load_model_checkpoint(model, model_dir, device, hps)
model = Cifar10Wrapper(model).to(device)

msda_config = rh.create_msda_config(hps)

#load dataset
od_bs = int(hps.od_bs_factor * hps.bs)

id_config = {}
if hps.dataset == 'cifar10':
    train_loader = dl.get_CIFAR10(train=True, batch_size=hps.bs, augm_type=hps.augm, size=img_size,
                                  config_dict=id_config)
elif hps.dataset == 'semi-cifar10':
    train_loader = dl.get_CIFAR10_ti_500k(train=True, batch_size=hps.bs, augm_type=hps.augm, fraction=0.7,
                                          size=img_size,
                                          config_dict=id_config)
else:
    raise ValueError(f'Dataset {hps.datset} not supported')

if hps.train_type.lower() in ['ceda', 'acet', 'advacet', 'tradesacet', 'tradesceda']:
    od_config = {}
    loader_config = {'ID config': id_config, 'OD config': od_config}

    if hps.od_dataset == 'tinyImages':
        tiny_train = dl.get_80MTinyImages(batch_size=od_bs, augm_type=hps.augm, num_workers=1, size=img_size,
                                          exclude_cifar=hps.exclude_cifar, exclude_cifar10_1=hps.exclude_cifar, config_dict=od_config)
    elif hps.od_dataset == 'cifar100':
        tiny_train = dl.get_CIFAR100(train=True, batch_size=od_bs, shuffle=True, augm_type=hps.augm,
                                     size=img_size, config_dict=od_config)
    elif hps.od_dataset == 'openImages':
        tiny_train = dl.get_openImages('train', batch_size=od_bs, shuffle=True, augm_type=hps.augm, size=img_size, exclude_dataset=None, config_dict=od_config)
else:
    loader_config = {'ID config': id_config}

test_loader = dl.get_CIFAR10(train=False, batch_size=hps.bs, augm_type='none', size=img_size)

scheduler_config, optimizer_config = rh.create_optim_scheduler_swa_configs(hps)
id_attack_config, od_attack_config = rh.create_attack_config(hps, 'cifar10')
trainer = rh.create_trainer(hps, model, optimizer_config, scheduler_config, device, num_classes,
                            model_dir, log_dir, msda_config=msda_config, model_config=model_config,
                            id_attack_config=id_attack_config, od_attack_config=od_attack_config)
##DEBUG:
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

# run training
if trainer.requires_out_distribution():
    train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader,
                                                              out_distribution_loader=tiny_train)
    trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                  optim_state_dict=optim_state_dict, device_ids=device_ids)
else:
    train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader)
    trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                  optim_state_dict=optim_state_dict, device_ids=device_ids)
