import matplotlib as mpl

mpl.use('Agg')

import torch
import torch.nn as nn

from utils.model_normalization import RestrictedImageNetWrapper
import utils.models.model_factory_224 as factory
import utils.run_file_helpers as rh
import utils.datasets as dl
import os
import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser.add_argument('--net', type=str, default='ResNet50', help='Resnet18, 34 or 50, WideResNet28')
parser.add_argument('--img_size', type=int, default='224', help='Img out_size')
parser.add_argument('--dataset', type=str, default='restrictedImagenet', help='Dataset')
parser.add_argument('--od_dataset', type=str, default='restrictedImagenetOD',
                    help=('restrictedImagenetOD or openimages'))

rh.parser_add_commons(parser)
rh.parser_add_adversarial_commons(parser)
rh.parser_add_adversarial_norms(parser, 'restrictedImagenet')

hps = parser.parse_args()
#
if len(hps.gpu) == 0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
elif len(hps.gpu) == 1:
    device = torch.device('cuda:' + str(hps.gpu[0]))
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))


# Load model
model_root_dir = 'RestrictedImageNetModels'
logs_root_dir = 'RestrictedImageNetLogs'

img_size = hps.img_size
num_classes = 9
model, model_name, model_config = factory.build_model(hps.net, num_classes)
model_dir = os.path.join(model_root_dir, model_name)
log_dir = os.path.join(logs_root_dir, model_name)
start_epoch, optim_state_dict = rh.load_model_checkpoint(model, model_dir, device, hps)
model = RestrictedImageNetWrapper(model).to(device)
if len(hps.gpu) > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

msda_config = rh.create_msda_config(hps)

# load dataset
od_bs = int(hps.od_bs_factor * hps.bs)
balanced = False

if hps.train_type.lower() in ['ceda', 'acet', 'advacet', 'tradesacet', 'tradesceda']:
    id_config = {}
    od_config = {}
    loader_config = {'ID config': id_config, 'OD config': od_config}

    if hps.dataset == 'restrictedImagenet':
        train_loader = dl.get_restrictedImageNet(train=True, batch_size=hps.bs, shuffle=True, augm_type=hps.augm,
                                                 size=img_size, balanced=balanced, config_dict=id_config)
    else:
        raise ValueError(f'Dataset {hps.datset} not supported')

    if hps.od_dataset == 'openimages':
        od_loader = dl.get_openImages('train', batch_size=od_bs, shuffle=True, augm_type=hps.augm, size=img_size,
                                      config_dict=od_config)
    elif hps.od_dataset == 'restrictedImagenetOD':
        od_loader = dl.get_restrictedImageNetOD(train=True, batch_size=od_bs, shuffle=True,
                                                augm_type=hps.augm, size=img_size, config_dict=od_config)
    else:
        raise ValueError()
else:
    id_config = {}
    loader_config = {'ID config': id_config}

    if hps.dataset == 'restrictedImagenet':
        train_loader = dl.get_restrictedImageNet(train=True, batch_size=hps.bs, shuffle=True, augm_type=hps.augm,
                                                 size=img_size, balanced=balanced, config_dict=id_config)
    else:
        raise ValueError(f'Dataset {hps.datset} not supported')

test_loader = dl.get_restrictedImageNet(train=False, batch_size=hps.bs, augm_type='none', size=img_size, balanced=False)

scheduler_config, optimizer_config = rh.create_optim_scheduler_swa_configs(hps)

trainer = rh.create_trainer(hps, model, optimizer_config, scheduler_config, device, num_classes,
                            model_dir, log_dir, msda_config, model_config=model_config)

##DEBUG:
# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

# run training
if trainer.requires_out_distribution():
    train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader,
                                                              out_distribution_loader=od_loader)
    trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                  optim_state_dict=optim_state_dict)
else:
    train_loaders, test_loaders = trainer.create_loaders_dict(train_loader, test_loader=test_loader)
    trainer.train(train_loaders, test_loaders, loader_config=loader_config, start_epoch=start_epoch,
                  optim_state_dict=optim_state_dict)
