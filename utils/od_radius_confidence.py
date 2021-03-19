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
import utils.datasets as dl
import utils.adversarial_attacks as at
from utils.train_types.train_loss import MaxConfidenceLoss, NegativeWrapper


def od_radius_confidence(model_descriptions, radii, plot_radii, dataloader, bs, datapoints, class_labels, device, eval_dir, dataset, img_size=32, norm='L2', steps=500, stepsize=0.1, device_ids=None):
    model_radii_mmc = torch.zeros(len(model_descriptions), len(radii))
    num_classes = len(class_labels)

    num_batches = int(np.ceil(datapoints / bs))

    data_iterator = iter(dataloader)

    data_batches = []
    dir = f'{eval_dir}/Original/ODWorstCaseConfidence_new/'
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    example_idx = 0
    samples_collected = 0
    for _ in range(num_batches):
        data = next(data_iterator)[0]
        if (samples_collected + data.shape[0]) > datapoints:
            raise NotImplementedError()

        samples_collected += data.shape[0]
        data_batches.append(data)
        # for img_idx in range(ref_data.shape[0]):
        #     img = ref_data[img_idx,:]
        #     file_pre = f'{dir}{example_idx}_gt.png'
        #     img_pil = transforms.ToPILImage()(img)
        #     img_pil.save(file_pre)
        #     example_idx += 1

    with torch.no_grad():
        for model_idx, (type, folder, checkpoint, temperature, temp) in enumerate(model_descriptions):
            dir = f'{eval_dir}/{folder}_{checkpoint}/ODWorstCaseConfidence_new/'
            print(f'Starting OD process: {folder}')
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
            indiv_dir = f'{dir}SingleImages/'
            pathlib.Path(indiv_dir).mkdir(parents=True, exist_ok=True)

            model = load_model(type, folder, checkpoint, temperature, device, load_temp=temp, dataset=dataset)
            if device_ids is not None and len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)
            model.eval()


            imgs = torch.zeros((len(radii), datapoints, 3, img_size, img_size), dtype=torch.float32)
            predictions = torch.zeros((len(radii), datapoints), dtype=torch.long)
            confidences = torch.zeros((len(radii), datapoints), dtype=torch.float32)

            datapoint_idx = 0
            for batch_idx, data in enumerate(data_batches):
                data = data.to(device)
                _, model_predictions = torch.max(model(data), dim=1)
                #attack current ref_data for all radi
                for radius_idx, radius in enumerate(radii):
                    if radius > 1e-8:
                        step_multiplier = 5
                        eps = radii[radius_idx]

                        if temperature is not None and temperature < 0.2:
                            loss = 'logit_max_target'
                            raise NotImplementedError()
                        else:
                            loss = 'conf'

                        # att = at.APGDAttack(density_model, num_classes, n_restarts=1, n_iter=100 * step_multiplier,
                        #                   eps=eps, norm='L2', loss=loss, eot_iter=1)

                        att = at.ArgminPGD(eps, steps, stepsize, num_classes, norm=norm, loss=loss, model=model)
                        adv_samples = att.perturb(data, model_predictions, targeted=True).detach()
                    else:
                        adv_samples = data

                    max_conf, predicted_class = torch.max(F.softmax(model(adv_samples), dim=1), dim=1)

                    imgs[radius_idx, datapoint_idx:(datapoint_idx + data.shape[0]), :] = adv_samples.detach().cpu()
                    predictions[radius_idx, datapoint_idx:(datapoint_idx + data.shape[0])] = predicted_class.detach().cpu()
                    confidences[radius_idx, datapoint_idx:(datapoint_idx + data.shape[0])] = max_conf.detach().cpu()

                    # #individual imgs
                    # for img_idx in range(adv_samples.shape[0]):
                    #     img_cpu = adv_samples[img_idx,:].detach().cpu()
                    #     img_pil = transforms.ToPILImage()(img_cpu)
                    #     file = f'{indiv_dir}batch_{batch_idx}_img_{img_idx}_radius_{radius_idx}-{radius}.png'
                    #     img_pil.save(file)

                #jump to next ref_data batch
                datapoint_idx += adv_samples.shape[0]

            model_radii_mmc[model_idx, :] = torch.mean(confidences, dim=1)

            print(f'{folder}')
            print(f'Max confs: {model_radii_mmc[model_idx, :]}')

            # plt combo plot
            num_radii_to_plot = 0
            for plot_radius in plot_radii:
                if plot_radius:
                    num_radii_to_plot += 1

            for img_idx in range(datapoints):
                scale_factor = 1.5
                fig, axs = plt.subplots(1, num_radii_to_plot, figsize=(scale_factor * num_radii_to_plot, 1.3 * scale_factor))

                col_idx = 0
                for radius_idx in range(len(radii)):
                    if plot_radii[radius_idx]:
                        axs[col_idx].axis('off')
                        axs[col_idx].title.set_text(
                            f'{class_labels[predictions[radius_idx, img_idx]]} - {confidences[radius_idx, img_idx]:.2f}')
                        img_cpu = imgs[radius_idx, img_idx, :].permute(1, 2, 0)
                        axs[col_idx].imshow(img_cpu, interpolation='lanczos')
                        col_idx += 1

                plt.tight_layout()
                fig.savefig(f'{dir}img_{img_idx}.png')
                fig.savefig(f'{dir}img_{img_idx}.pdf')
                plt.close(fig)

            # # plt combo plot with diff
            # for img_idx in range(datapoints):
            #     scale_factor = 1.5
            #     fig, axs = plt.subplots(2, num_radii_to_plot, figsize=(scale_factor * num_radii_to_plot, 2 * 1.3 * scale_factor))
            #
            #     orig_img = imgs[0, img_idx, :].permute(1, 2, 0)
            #
            #     col_idx = 0
            #     for radius_idx in range(len(radii)):
            #         if plot_radii[radius_idx]:
            #             axs[0, col_idx].axis('off')
            #             axs[0, col_idx].title.set_text(
            #                 f'{class_labels[predictions[radius_idx, img_idx]]} - {confidences[radius_idx, img_idx]:.2f}')
            #             img_cpu = imgs[radius_idx, img_idx, :].permute(1, 2, 0)
            #             axs[0, col_idx].imshow(img_cpu, interpolation='lanczos')
            #
            #             amplification = 5
            #
            #             diff = torch.sum((orig_img - img_cpu) ** 2, dim=2, keepdim=True)
            #             diff_total = torch.sum(diff.view(diff.shape[0], -1))[..., None, None, None]
            #             diff_normalized = torch.sqrt(diff) / torch.sqrt(diff_total)
            #             diff_normalized = torch.cat([diff_normalized, diff_normalized, diff_normalized], dim=2)
            #             diff_normalized = torch.clamp(amplification * diff_normalized, 0, 1)
            #             axs[1, col_idx].axis('off')
            #             axs[1, col_idx].imshow(diff_normalized, interpolation='lanczos')
            #             col_idx += 1
            #
            #     plt.tight_layout()
            #     fig.savefig(f'{dir}img_{img_idx}_diff.png')
            #     fig.savefig(f'{dir}img_{img_idx}_diff.pdf')
            #     plt.close(fig)

            # # animated gif parts
            # rows = int(np.sqrt(datapoints))
            # cols = int(np.ceil(datapoints / rows))
            # scale_factor = 4
            #
            # for radius_idx, radius in enumerate(radii):
            #     if plot_radii[radius_idx]:
            #         fig, axs = plt.subplots(rows, cols, figsize=(scale_factor * cols, scale_factor * rows))
            #         fig.suptitle(f'Radius: {radius}')
            #         for img_idx in range(datapoints):
            #             row_idx = int(img_idx / cols)
            #             col_idx = int(img_idx % cols)
            #
            #             img_cpu = imgs[radius_idx, img_idx, :].permute(1, 2, 0)
            #
            #             axs[row_idx, col_idx].axis('off')
            #             # axs[row_idx, col_idx].title.set_text(
            #             #     f'{class_labels[predictions[radius_idx, img_idx]]} - {confidences[radius_idx, img_idx]:.2f} - r {radii[radius_idx]:.2f}')
            #             axs[row_idx, col_idx].imshow(img_cpu, interpolation='lanczos')
            #
            #         fig.savefig(f'{dir}img_gif_part_{radius_idx}.png')
            #         fig.savefig(f'{dir}img_gif_part_{radius_idx}.pdf')
            #         plt.close(fig)

        #torch.save(model_radii_mmc, f'{eval_dir}OD_model_radii_mmc.pt')
    return model_radii_mmc