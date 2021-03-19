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
from PIL import Image
from utils.load_trained_model import load_model
import utils.dataloaders as dl
from utils.train_types.train_loss import MaxConfidenceLoss, NegativeWrapper
from auto_attack.autopgd_pt import APGDAttack_singlestepsize as APGDAttack
from tqdm import trange

def generate_all_classes(model_descriptions, radii, dataloader, dataloader_name, bs, datapoints, class_labels, device, eval_dir, dataset, device_ids=None):
    num_classes = len(class_labels)
    model_radii_confs = torch.zeros(len(model_descriptions), len(radii), 2)

    indiv_batches = []
    data_collected = 0
    for batch_idx, (source_data, _) in enumerate(dataloader):
        num_to_collect = min(source_data.shape[0], datapoints - data_collected)

        if num_to_collect == 0:
            break

        source_data = source_data[:num_to_collect,:]
        indiv_batches.append(source_data)
        data_collected += num_to_collect

    source_data = torch.cat(indiv_batches, dim=0)

    img_dim = source_data.shape[1:]
    datapoints = source_data.shape[0]

    for model_idx in trange(len(model_descriptions), desc='Models progress'):
        type, folder, checkpoint, temperature, temp = model_descriptions[model_idx]
        dir = f'{eval_dir}/{folder}_{checkpoint}/Generative_{dataloader_name}/'
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        model = load_model(type, folder, checkpoint, temperature, device, load_temp=temp, dataset=dataset)
        if device_ids is not None and len(device_ids) > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model.eval()

        n_batches = int(np.ceil(datapoints / bs))

        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min(datapoints, (batch_idx + 1) * bs)

            data = source_data[start_idx:end_idx,:].to(device)
            orig_out = model(data)
            orig_conf, orig_pred = torch.max(torch.softmax(orig_out, dim=1), 1)

            # creates batches that have dimension batch_size * num_classes
            stacked_data = data.repeat(num_classes, 1, 1, 1)
            stacked_targets = data.new_empty(data.shape[0] * num_classes, dtype=torch.long)
            for c in range(num_classes):
                stacked_targets[(c * num_classes):((c + 1) * num_classes)] = c

            for radius_idx, radius in enumerate(radii):
                conf_targeted_attacked = torch.zeros((num_classes, data.shape[0]))
                pred_targeted_attacked =  torch.zeros((num_classes, data.shape[0]), dtype=torch.long)
                data_targeted_attacked = torch.zeros((num_classes, data.shape[0]) + data.shape[1:])
                if temperature < 0.2:
                    loss = 'logit_max_target'
                else:
                    loss = 'conf_target'

                step_multiplier = 5
                att = APGDAttack(model, n_restarts=1, n_iter=100 * step_multiplier, n_iter_2=22 * step_multiplier,
                                 n_iter_min=6 * step_multiplier, size_decr=3,
                                 eps=radius, show_loss=False, norm='L2', loss=loss, eot_iter=1,
                                 thr_decr=.75, seed=0, normalize_logits=True,
                                 show_acc=False)

                adv_samples = att.perturb(stacked_data, stacked_targets)[1]

                out = model(adv_samples)
                conf, pred = torch.max(torch.softmax(out, dim=1), 1)

                # might be possible to reshape this?

                for c in range(num_classes):
                    conf_targeted_attacked[c, :] = conf[(c * num_classes):((c + 1) * num_classes)]
                    pred_targeted_attacked[c, :] = pred[(c * num_classes):((c + 1) * num_classes)]
                    data_targeted_attacked[c, :] = adv_samples[(c * num_classes):((c + 1) * num_classes),:]

                fig, ax = plt.subplots(data.shape[0], num_classes + 1, figsize=(20, 22))

                for i in range(data.shape[0]):
                    # plot original
                    img_orig = data[i, :].detach().cpu().permute(1, 2, 0)
                    if img_orig.shape[2] == 1:
                        img_orig.squeeze_()
                        ax[i, 0].imshow(img_orig, cmap='gray')
                    else:
                        ax[i, 0].imshow(img_orig)

                    ax[i, 0].axis('off')
                    ax[i, 0].title.set_text('{:d} - {:0.2f}'.format(orig_pred[i].item(),
                                                                    orig_conf[i].item()))

                    for j in range(num_classes):
                        img_ij = data_targeted_attacked[j, i, :].permute(1, 2, 0)
                        if img_ij.shape[2] == 1:
                            img_ij.squeeze_()
                            ax[i, j + 1].imshow(img_ij, cmap='gray')
                        else:
                            ax[i, j + 1].imshow(img_ij)

                        ax[i, j + 1].axis('off')
                        ax[i, j + 1].title.set_text(
                            '{:d} - {:0.2f}'.format(pred_targeted_attacked[j, i].item(), conf_targeted_attacked[j, i].item()))

                plt.tight_layout()
                fig.savefig(f'{dir}{batch_idx}_generative_radius{radius:.2f}.png')
                fig.savefig(f'{dir}{batch_idx}_generative_radius{radius:.2f}.pdf')
                plt.close(fig)
