import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pathlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import utils.adversarial_attacks as aa
from autoattack import AutoAttack

from utils.load_trained_model import load_model
import utils.datasets as dl

model_descriptions = [
    ('WideResNet34x10', 'cifar10_pgd', 'best_avg', None, False),
    ('WideResNet34x10', 'cifar10_apgd', 'best_avg', None, False),
    ('WideResNet34x10', 'cifar10_500k_pgd', 'best_avg', None, False),
    ('WideResNet34x10', 'cifar10_500k_apgd', 'best_avg', None, False),
]


parser = argparse.ArgumentParser(description='Parse arguments.', prefix_chars='-')

parser.add_argument('--gpu','--list', nargs='+', default=[0],
                    help='GPU indices, if more than 1 parallel modules will be called')

hps = parser.parse_args()

if len(hps.gpu)==0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
    num_devices = 1
elif len(hps.gpu)==1:
    device_ids = [int(hps.gpu[0])]
    device = torch.device('cuda:' + str(hps.gpu[0]))
    num_devices = 1
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))
    num_devices = len(device_ids)

L2 = True
LINF = False

ROBUSTNESS_DATAPOINTS = 10_000
dataset = 'cifar10'

bs = 500 * num_devices

print(f'Testing on {ROBUSTNESS_DATAPOINTS} points')

for model_idx, (type, folder, checkpoint, temperature, temp) in enumerate(model_descriptions):
    model = load_model(type, folder, checkpoint,
                       temperature, device, load_temp=temp, dataset=dataset)
    model.to(device)

    if len(hps.gpu) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    model.eval()
    print(f'\n\n{folder} {checkpoint}\n ')

    if dataset == 'cifar10':
        dataloader = dl.get_CIFAR10(False, batch_size=bs, augm_type='none')
    elif dataset == 'cifar100':
        dataloader = dl.get_CIFAR100(False, batch_size=bs, augm_type='none')
    else:
        raise NotImplementedError()

    acc = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            _, pred = torch.max(out, dim=1)
            acc += torch.sum(pred == target).item() / len(dataloader.dataset)

    print(f'Clean accuracy {acc}')

    if dataset == 'cifar10':
        dataloader = dl.get_CIFAR10(False, batch_size=ROBUSTNESS_DATAPOINTS, augm_type='none')
    elif dataset == 'cifar100':
        dataloader = dl.get_CIFAR100(False, batch_size=ROBUSTNESS_DATAPOINTS, augm_type='none')
    else:
        raise NotImplementedError()
    
    data_iterator = iter(dataloader)
    ref_data, target = next(data_iterator)

    if L2:
        print('Eps: 0.5')

        attack = AutoAttack(model, device=device, norm='L2', eps=0.5, verbose=True)
        attack.run_standard_evaluation(ref_data, target, bs=bs)

        # print('Eps: 1.0')
        # attack = AutoAttack(model, device=device, norm='L2', eps=1.0, attacks_to_run=attacks_to_run,verbose=True)
        # attack.run_standard_evaluation(ref_data, target, bs=bs)
    if LINF:
        print('Eps: 8/255')
        attack = AutoAttack(model,  device=device, norm='Linf', eps=8./255.,verbose=True)
        attack.run_standard_evaluation(ref_data, target, bs=bs)

