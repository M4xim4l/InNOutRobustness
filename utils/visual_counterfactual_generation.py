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

from utils.load_trained_model import load_model
import utils.adversarial_attacks as at
from utils.train_types.train_loss import TrainLoss
from tqdm import trange
from time import sleep
import utils.resize_right as resize_right

def _find_wrong_examples(model_descriptions, dataloader, num_examples, class_labels, device, eval_dir, dataset, device_ids=None):
    num_classes = len(class_labels)
    num_datapoints = len(dataloader.dataset)

    batch_shape = next(iter(dataloader))[0].shape

    imgs = torch.zeros((num_datapoints, batch_shape[1], batch_shape[2], batch_shape[3]), dtype=torch.float)

    targets = torch.zeros(num_datapoints, dtype=torch.long)
    predictions = torch.zeros((num_datapoints, len(model_descriptions)), dtype=torch.long)
    confidences = torch.zeros((num_datapoints, len(model_descriptions), num_classes), dtype=torch.float32)
    failure = torch.zeros((num_datapoints, len(model_descriptions)), dtype=torch.bool)

    with torch.no_grad():
        for model_idx in trange(len(model_descriptions), desc='Models progress'):
            type, folder, checkpoint, temperature, temp = model_descriptions[model_idx]
            model = load_model(type, folder, checkpoint, temperature, device, load_temp=temp, dataset=dataset)
            # search failure cases
            if device_ids is not None and len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)
            model.eval()

            datapoint_idx = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    if model_idx == 0:
                        imgs[datapoint_idx:(datapoint_idx+data.shape[0]),:] = data
                        targets[datapoint_idx:(datapoint_idx + data.shape[0])] = target

                    data, target = data.to(device), target.to(device)

                    orig_out = model(data)
                    orig_confidences = torch.softmax(orig_out, dim=1)
                    _, orig_pred = torch.max(orig_confidences, 1)
                    correct = orig_pred.eq(target)

                    predictions[datapoint_idx:(datapoint_idx+data.shape[0]), model_idx] = orig_pred.detach().cpu()
                    confidences[datapoint_idx:(datapoint_idx+data.shape[0]), model_idx] = orig_confidences.detach().cpu()
                    failure[datapoint_idx:(datapoint_idx+data.shape[0]), model_idx] = ~correct

                    datapoint_idx += data.shape[0]

        all_model_failure = torch.sum(failure, dim=1) >= len(model_descriptions)
        all_model_failure_idcs = torch.nonzero(all_model_failure, as_tuple=False).squeeze()

        examples_found = min(all_model_failure_idcs.shape[0], num_examples)
        all_model_failure_idcs = all_model_failure_idcs[:examples_found]

        model_failure_examples = imgs[all_model_failure_idcs,:]
        model_failure_targets = targets[all_model_failure_idcs]
        all_model_failure_predictions = predictions[all_model_failure_idcs,:]
        all_model_failure_confidences = confidences[all_model_failure_idcs,:]

        print(f'Found {examples_found} out of {num_examples} falsely classified images')

        #save misclassified images
        dir = f'{eval_dir}/Original/VisualCounterfactuals/'
        indiv_dir = f'{dir}SingleImages/'
        pathlib.Path(indiv_dir).mkdir(parents=True, exist_ok=True)

        # plot individual gt imgs
        for example_idx in range(examples_found):
            img = model_failure_examples[example_idx, :]
            target = model_failure_targets[example_idx]
            filename = f'{indiv_dir}{example_idx}_gt.png'
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(filename)

            filename = f'{indiv_dir}{example_idx}_gt.txt'
            with open(filename, 'w') as fp:
                fp.write(f'{class_labels[target]}')

        return model_failure_examples, model_failure_targets, \
               all_model_failure_predictions, all_model_failure_confidences


def _plot_counterfactuals(dir, examples_found, gt_conf_imgs, false_conf_imgs, folder, checkpoint,
                          model_failure_examples,
                          model_failure_targets, model_failure_predictions, false_conf_confidences,
                          model_failure_confidences, gt_conf_confidences, radii, class_labels):
    # plot last images individually
    indiv_dir = f'{dir}SingleImages/'
    pathlib.Path(indiv_dir).mkdir(parents=True, exist_ok=True)

    for example_idx in range(examples_found):
        for i in range(2):
            if i == 0:
                img = gt_conf_imgs[-1, example_idx, :]
                filename = f'{indiv_dir}{example_idx}_gt.png'
            else:
                img = false_conf_imgs[-1, example_idx, :]
                filename = f'{indiv_dir}{example_idx}_predicted.png'
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(filename)

    # plot without difference
    scale_factor = 1.5
    for example_idx in trange(examples_found, desc=f'{folder} {checkpoint} - Image write'):
        ex_target = model_failure_targets[example_idx]
        ex_pred = model_failure_predictions[example_idx]

        fig, ax = plt.subplots(2, len(radii) + 1, figsize=(scale_factor * (len(radii) + 1), 2 * 1.3 * scale_factor))

        # plot original:
        ax[0, 0].axis('off')
        ax[0, 0].title.set_text(
            f'{class_labels[ex_target]}: {model_failure_confidences[example_idx, ex_target]:.2f}\n{class_labels[ex_pred]}:'
            f' {model_failure_confidences[example_idx, ex_pred]:.2f}')
        ex_img_original = model_failure_examples[example_idx, :].permute(1, 2, 0).cpu().detach()
        ax[0, 0].imshow(ex_img_original, interpolation='lanczos')
        ax[1, 0].axis('off')

        for i in range(2):
            for j in range(len(radii)):
                if i == 0:
                    img = gt_conf_imgs[j, example_idx, :].permute(1, 2, 0)
                    target_conf = gt_conf_confidences[j, example_idx, :][ex_target]
                    false_conf = gt_conf_confidences[j, example_idx, :][ex_pred]
                else:
                    img = false_conf_imgs[j, example_idx, :].permute(1, 2, 0)
                    target_conf = false_conf_confidences[j, example_idx, :][ex_target]
                    false_conf = false_conf_confidences[j, example_idx, :][ex_pred]

                ax[i, j + 1].axis('off')
                ax[i, j + 1].title.set_text(
                    f'{class_labels[ex_target]}: {target_conf:.2f}\n{class_labels[ex_pred]}: {false_conf:.2f}')
                ax[i, j + 1].imshow(img, interpolation='lanczos')

                amplification = 5

        plt.tight_layout()
        fig.savefig(f'{dir}{example_idx}.png')
        fig.savefig(f'{dir}{example_idx}.pdf')
        plt.close(fig)


def visual_counterfactuals(model_descriptions, radii, dataloader, bs, num_examples, class_labels, device, eval_dir,
                           dataset, norm='L2', steps=500, stepsize=0.1, pyramid_levels=1, device_ids=None):
    num_classes = len(class_labels)

    model_failure_examples, model_failure_targets, all_model_failure_predictions, all_model_failure_confidences = \
        _find_wrong_examples(model_descriptions, dataloader, num_examples, class_labels, device, eval_dir, dataset,
                             device_ids=device_ids)

    examples_found = len(model_failure_examples)
    with torch.no_grad():

        for model_idx in range(len(model_descriptions)):
            if np.isscalar(bs):
                model_bs = bs
            else:
                model_bs = bs[model_idx]

            type, folder, checkpoint, temperature, temp = model_descriptions[model_idx]
            print(f'{folder} {checkpoint} - bs {model_bs}')

            dir = f'{eval_dir}/{folder}_{checkpoint}/VisualCounterfactuals/'
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

            model = load_model(type, folder, checkpoint, temperature, device, load_temp=temp, dataset=dataset)
            if device_ids is not None and len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)
            model.eval()

            gt_conf_imgs = torch.zeros((len(radii),) + model_failure_examples.shape)
            false_conf_imgs = torch.zeros_like(gt_conf_imgs)

            gt_conf_confidences = torch.zeros((len(radii), num_examples, num_classes))
            false_conf_confidences = torch.zeros((len(radii), num_examples, num_classes))

            n_batches = int(np.ceil(examples_found / model_bs))

            for batch_idx in trange(n_batches, desc=f'{folder} {checkpoint} - Batches progress'):
                sleep(0.1)
                batch_start_idx = batch_idx * model_bs
                batch_end_idx = min(examples_found, (batch_idx + 1) * model_bs)

                batch_data = model_failure_examples[batch_start_idx:batch_end_idx, :]
                batch_gt_targets = model_failure_targets[batch_start_idx:batch_end_idx]
                batch_failure_predictions = all_model_failure_predictions[batch_start_idx:batch_end_idx, model_idx]

                for radius_idx in range(len(radii)):
                    eps = radii[radius_idx]

                    if temperature is not None and temperature < 0.2:
                        loss = 'logit_max_target'
                        raise NotImplementedError()
                    else:
                        loss = 'confdiff'  # NegativeWrapper(ConfidenceLoss())

                    # att = at.APGDAttack(density_model, num_classes, n_restarts=1, n_iter=100 * step_multiplier,
                    #                   eps=eps, norm=norm, loss=loss, eot_iter=1)

                    if pyramid_levels > 1:
                        batch_gt_targets = batch_gt_targets.to(device)
                        batch_failure_predictions = batch_failure_predictions.to(device)

                        with torch.no_grad():
                            gt_exs = resize_right.resize(batch_data,
                                                         2 ** (-(pyramid_levels - 1))).contiguous().to(device)
                            false_exs = gt_exs.clone().detach().to(device)


                        for lvl in range(pyramid_levels):
                            scale_factor_lvl = 2 ** (-(pyramid_levels - 1 - lvl))

                            if norm == 'l1':
                                #scale factor is per spatial dimension, thus square of total scaling factor
                                # l2 norm scales linearly in total pixels
                                lvl_eps = eps * scale_factor_lvl**2
                                lvl_stepsize = stepsize  * scale_factor_lvl**2
                            elif norm == 'l2':
                                #scale factor is per spatial dimension, thus square of total scaling factor
                                # l2 norm scales with sqrt in total pixels
                                lvl_eps = eps * scale_factor_lvl
                                lvl_stepsize = stepsize  * scale_factor_lvl

                            else:
                                lvl_eps = eps

                            att = at.ArgminPGD(lvl_eps, steps, lvl_stepsize, num_classes, norm=norm, loss=loss, momentum=0,
                                               model=model)

                            if lvl > 0:
                                with torch.no_grad():
                                    gt_exs = resize_right.resize(gt_exs, 2).contiguous().to(
                                        device)
                                    false_exs = resize_right.resize(false_exs, 2).contiguous().to(
                                        device)

                            with torch.no_grad():
                                if scale_factor_lvl < 1:
                                    lvl_batch_data = resize_right.resize(batch_data, scale_factor_lvl).contiguous().to(
                                        device)
                                else:
                                    lvl_batch_data = batch_data.to(device)

                            gt_exs = att.perturb(lvl_batch_data, batch_gt_targets, targeted=True, x_init=gt_exs).detach()
                            false_exs = att.perturb(lvl_batch_data, batch_failure_predictions, targeted=True, x_init=false_exs).detach()
                    else:
                        batch_data = batch_data.to(device)
                        batch_gt_targets = batch_gt_targets.to(device)
                        batch_failure_predictions = batch_failure_predictions.to(device)

                        att = at.ArgminPGD(eps, steps, stepsize, num_classes, norm=norm, loss=loss, momentum=0,
                                           model=model)

                        gt_exs = att.perturb(batch_data, batch_gt_targets, targeted=True).detach()
                        false_exs = att.perturb(batch_data, batch_failure_predictions, targeted=True).detach()

                    gt_conf_imgs[radius_idx, batch_start_idx:batch_end_idx, :] = gt_exs.cpu().detach()
                    false_conf_imgs[radius_idx, batch_start_idx:batch_end_idx, :] = false_exs.cpu().detach()

                    gt_out = model(gt_exs)
                    gt_confs = torch.softmax(gt_out, dim=1)
                    gt_conf_confidences[radius_idx, batch_start_idx:batch_end_idx, :] = gt_confs.cpu().detach()

                    false_out = model(false_exs)
                    false_confs = torch.softmax(false_out, dim=1)
                    false_conf_confidences[radius_idx, batch_start_idx:batch_end_idx, :] = false_confs.cpu().detach()

            model_failure_confidences = all_model_failure_confidences[:, model_idx]
            model_failure_predictions = all_model_failure_predictions[:, model_idx]

            _plot_counterfactuals(dir, examples_found, gt_conf_imgs, false_conf_imgs, folder, checkpoint,
                                  model_failure_examples,
                                  model_failure_targets, model_failure_predictions, false_conf_confidences,
                                  model_failure_confidences, gt_conf_confidences, radii, class_labels)

class MinusLogPosterior(TrainLoss):
    def __init__(self, density_model):
        super().__init__('MinusLogPosterior', expected_format='log_probabilities')
        self.density_model = density_model

    #expects the logits of the density_model responsible for p(y|x) in density_model out
    def forward(self, data, model_out, orig_data, y, reduction='mean'):
        log_prior = self.density_model(data, None)
        prep_out = self._prepare_input(model_out)
        log_likelihood = prep_out[torch.arange(0, prep_out.shape[0]), y]
        return -(log_prior + log_likelihood)

def visual_counterfactuals_plus_prior(model_descriptions, density_model, radii, dataloader, bs, num_examples,
                                      class_labels, device, eval_dir,
                           dataset, norm='L2', steps=500, stepsize=0.1, pyramid_levels=1, device_ids=None):
    num_classes = len(class_labels)

    model_failure_examples, model_failure_targets, all_model_failure_predictions, all_model_failure_confidences = \
        _find_wrong_examples(model_descriptions, dataloader, num_examples, class_labels, device, eval_dir, dataset,
                             device_ids=device_ids)

    examples_found = len(model_failure_examples)
    with torch.no_grad():

        for model_idx in range(len(model_descriptions)):
            if np.isscalar(bs):
                model_bs = bs
            else:
                model_bs = bs[model_idx]

            type, folder, checkpoint, temperature, temp = model_descriptions[model_idx]
            print(f'{folder} {checkpoint} - bs {model_bs}')

            dir = f'{eval_dir}/{folder}_{checkpoint}/VisualCounterfactuals/'
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

            model = load_model(type, folder, checkpoint, temperature, device, load_temp=temp, dataset=dataset)
            if device_ids is not None and len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)
            model.eval()

            gt_conf_imgs = torch.zeros((len(radii),) + model_failure_examples.shape)
            false_conf_imgs = torch.zeros_like(gt_conf_imgs)

            gt_conf_confidences = torch.zeros((len(radii), num_examples, num_classes))
            false_conf_confidences = torch.zeros((len(radii), num_examples, num_classes))

            n_batches = int(np.ceil(examples_found / model_bs))



            for batch_idx in trange(n_batches, desc=f'{folder} {checkpoint} - Batches progress'):
                sleep(0.1)
                batch_start_idx = batch_idx * model_bs
                batch_end_idx = min(examples_found, (batch_idx + 1) * model_bs)

                batch_data = model_failure_examples[batch_start_idx:batch_end_idx, :]
                batch_gt_targets = model_failure_targets[batch_start_idx:batch_end_idx]
                batch_failure_predictions = all_model_failure_predictions[batch_start_idx:batch_end_idx, model_idx]

                loss = MinusLogPosterior(density_model)

                for radius_idx in range(len(radii)):
                    eps = radii[radius_idx]
                    if pyramid_levels > 1:
                        gt_exs = resize_right.resize(batch_data, 2 ** (-(pyramid_levels - 1))).detach().to(device)
                        false_exs = gt_exs.clone().detach().to(device)
                        att = at.ArgminPGD(eps, steps, stepsize, num_classes, norm=norm, loss=loss, momentum=0,
                                           model=model)

                        for lvl in range(pyramid_levels):
                            if lvl > 0:
                                gt_exs = resize_right.resize(gt_exs, 2).detach().to(
                                    device)
                                false_exs = resize_right.resize(false_exs, 2).detach().to(
                                    device)

                            gt_exs = att.perturb(gt_exs, batch_gt_targets, targeted=True).detach()
                            false_exs = att.perturb(false_exs, batch_failure_predictions, targeted=True).detach()
                    else:
                        batch_data = batch_data.to(device)
                        batch_gt_targets = batch_gt_targets.to(device)
                        batch_failure_predictions = batch_failure_predictions.to(device)

                        # att = at.ArgminPGD(eps, steps, stepsize, num_classes, norm=norm, loss=loss, momentum=0,
                        #                    model=model)
                        att = at.APGDAttack(model, num_classes, eps, steps, norm=norm, loss=loss)

                        gt_exs = att.perturb(batch_data, batch_gt_targets, targeted=True).detach()
                        false_exs = att.perturb(batch_data, batch_failure_predictions, targeted=True).detach()

                    gt_conf_imgs[radius_idx, batch_start_idx:batch_end_idx, :] = gt_exs.cpu().detach()
                    false_conf_imgs[radius_idx, batch_start_idx:batch_end_idx, :] = false_exs.cpu().detach()

                    gt_out = model(gt_exs)
                    gt_confs = torch.softmax(gt_out, dim=1)
                    gt_conf_confidences[radius_idx, batch_start_idx:batch_end_idx, :] = gt_confs.cpu().detach()

                    false_out = model(false_exs)
                    false_confs = torch.softmax(false_out, dim=1)
                    false_conf_confidences[radius_idx, batch_start_idx:batch_end_idx, :] = false_confs.cpu().detach()

            model_failure_confidences = all_model_failure_confidences[:, model_idx]
            model_failure_predictions = all_model_failure_predictions[:, model_idx]

            _plot_counterfactuals(dir, examples_found, gt_conf_imgs, false_conf_imgs, folder, checkpoint,
                                  model_failure_examples,
                                  model_failure_targets, model_failure_predictions, false_conf_confidences,
                                  model_failure_confidences, gt_conf_confidences, radii, class_labels)
