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

import utils.datasets as dl
from utils.visual_counterfactual_generation import visual_counterfactuals

parser = argparse.ArgumentParser(description='Parse arguments.', prefix_chars='-')

parser.add_argument('--gpu','--list', nargs='+', default=[0],
                    help='GPU indices, if more than 1 parallel modules will be called')

hps = parser.parse_args()


bs = 128
big_model_bs = 20

if len(hps.gpu)==0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
elif len(hps.gpu)==1:
    device_ids = None
    device = torch.device('cuda:' + str(hps.gpu[0]))
    bs = bs
    big_model_bs = big_model_bs
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))
    bs = bs * len(device_ids)
    big_model_bs = big_model_bs * len(device_ids)

model_descriptions = [
    ('WideResNet34x10', 'cifar10_pgd', 'best_avg', None, False),
    ('WideResNet34x10', 'cifar10_apgd', 'best_avg', None, False),
    ('WideResNet34x10', 'cifar10_500k_pgd', 'best_avg', None, False),
    ('WideResNet34x10', 'cifar10_500k_apgd', 'best_avg', None, False),
]

model_batchsize = bs * np.ones(len(model_descriptions), dtype=np.int)
num_examples = 100

dataloader = dl.get_CIFAR10(False, bs, augm_type='none')
num_datapoints = len(dataloader.dataset)

class_labels = dl.cifar.get_CIFAR10_labels()
eval_dir = 'Cifar10Eval/'

norm = 'l2'

if norm == 'l1':
    radii = np.linspace(15, 90, 6)
    visual_counterfactuals(model_descriptions, radii, dataloader, model_batchsize, num_examples, class_labels, device,
                           eval_dir, 'cifar10', norm='l1', stepsize=5, device_ids=device_ids)
else:
    radii = np.linspace(0.5, 3, 6)
    visual_counterfactuals(model_descriptions, radii, dataloader, model_batchsize, num_examples, class_labels, device, eval_dir, 'cifar10', device_ids=device_ids)
