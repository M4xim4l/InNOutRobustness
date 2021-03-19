from torchvision import transforms
import torch
from utils.datasets.autoaugment import SVHNPolicy, CIFAR10Policy
from utils.datasets.cutout import Cutout

SVHN_mean_int = ( int( 255 * 0.4377), int(255 * 0.4438), int(255 * 0.4728))
SVHN_mean = torch.tensor([0.4377, 0.4438, 0.4728])


def get_SVHN_augmentation(augm_type='none', size=32, config_dict=None):
    if augm_type == 'none':
        transform_list = []
    elif augm_type == 'default' or augm_type == 'default_cutout':
        transform_list = [
            transforms.RandomCrop(32, padding=4, fill=SVHN_mean_int),
        ]
    elif augm_type == 'autoaugment' or augm_type == 'autoaugment_cutout':
        transform_list = [
            transforms.RandomCrop(32, padding=4, fill=SVHN_mean_int),
            SVHNPolicy(fillcolor=SVHN_mean_int),
        ]
    elif augm_type == 'cifar_autoaugment' or augm_type == 'cifar_autoaugment_cutout':
        transform_list = [
            transforms.RandomCrop(32, padding=4, fill=SVHN_mean_int),
            CIFAR10Policy(fillcolor=SVHN_mean_int),
        ]
    else:
        raise ValueError()

    in_size = 32
    cutout_window = 16
    cutout_color = SVHN_mean

    if size != 32:
        if 'cutout' in augm_type:
            transform_list.append(transforms.Resize(size))
            transform_list.append(transforms.ToTensor())
            cutout_size = int(size / in_size * cutout_window)
            print(f'Relative Cutout window {cutout_window / in_size} - Absolute Cutout window: {cutout_size}')
            transform_list.append(Cutout(n_holes=1, length=cutout_size, fill_color=cutout_color))
        else:
            transform_list.append(transforms.Resize(size))
            transform_list.append(transforms.ToTensor())
    else:
        if 'cutout' in augm_type:
            cutout_size = cutout_window
            print(f'Relative Cutout window {cutout_size / in_size} - Absolute Cutout window: {cutout_size}')
            transform_list.append(transforms.ToTensor())
            transform_list.append(Cutout(n_holes=1, length=cutout_size, fill_color=cutout_color))
        else:
            transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)

    if config_dict is not None:
        config_dict['type'] = type
        config_dict['Input out_size'] = in_size
        config_dict['Output out_size'] = size
        if 'cutout' in augm_type:
            config_dict['Cutout out_size'] = cutout_size

    return transform