import torch
from torchvision import transforms

from utils.datasets.autoaugment import CIFAR10Policy, ImageNetPolicy
from utils.datasets.cutout import Cutout

CIFAR10_mean_int = ( int( 255 * 0.4913997551666284), int(255 * 0.48215855929893703), int(255 * 0.4465309133731618))
CIFAR10_mean = torch.tensor([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])

def get_cifar10_augmentation(type='default', cutout_window=16, out_size=32, in_size=32, magnitude_factor=1, config_dict=None):
    cutout_color = torch.tensor([0., 0., 0.])
    padding_size = 4 * int(in_size / 32)
    force_no_resize = False

    if type == 'none' or type is None:
        transform_list = []
    elif type == 'default' or type == 'default_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=CIFAR10_mean_int),
            transforms.RandomHorizontalFlip(),
        ]
        cutout_color = CIFAR10_mean
    elif type == 'madry' or type == 'madry_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=CIFAR10_mean_int),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
        ]
        cutout_color = CIFAR10_mean
    elif type == 'autoaugment' or type == 'autoaugment_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=CIFAR10_mean_int),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(fillcolor=CIFAR10_mean_int, magnitude_factor=magnitude_factor),
        ]
    elif type == 'in_autoaugment' or type == 'in_autoaugment_cutout':
        transform_list = [
            transforms.RandomCrop(in_size, padding=padding_size, fill=CIFAR10_mean_int),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(fillcolor=CIFAR10_mean_int),
        ]
        cutout_color = CIFAR10_mean
    elif type == 'big_transfer' or type == 'big_transfer_128':
        if type == 'big_transfer':
            if out_size != 480:
                print(f'Out out_size of {out_size} detected but Big Transfer is supposed to be used with 480')
                pre_crop_size = int(out_size * (512/480))
            else:
                pre_crop_size = 512
        else:
            if out_size != 128:
                print(f'Out out_size of {out_size} detected but Big Transfer 128 is supposed to be used with 128')
                pre_crop_size = int(out_size * (160 / 128))
            else:
                pre_crop_size = 160

        print(f'BigTransfer augmentation: Pre crop {pre_crop_size} - Out Size {out_size}')
        transform_list = [
            transforms.transforms.Resize((pre_crop_size, pre_crop_size)),
            transforms.transforms.RandomCrop((out_size, out_size)),
            transforms.transforms.RandomHorizontalFlip(),
        ]
        force_no_resize = True
    else:
        raise ValueError(f'augmentation type - {type} - not supported')

    if out_size != in_size and not force_no_resize:
        if 'cutout' in type:
            transform_list.append(transforms.Resize(out_size))
            transform_list.append(transforms.ToTensor())
            cutout_size = int(out_size / in_size * cutout_window)
            print(f'Relative Cutout window {cutout_window / in_size} - Absolute Cutout window: {cutout_size}')
            transform_list.append(Cutout(n_holes=1, length=cutout_size, fill_color=cutout_color))
        else:
            transform_list.append(transforms.Resize(out_size))
            transform_list.append(transforms.ToTensor())
    elif 'cutout' in type:
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
        config_dict['Output out_size'] = out_size
        config_dict['Magnitude factor'] = magnitude_factor
        if 'cutout' in type:
            config_dict['Cutout out_size'] = cutout_size

    return transform