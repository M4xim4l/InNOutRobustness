import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pathlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



def create_targeted_perturbation_matrix(model_list, attacks, device, dataloaders_list, num_classes, base_dir, batches=None):
    #create a targeted attack for each class in dataset starting from all datapoints for x batches
    for (dataloader_name, dataloader) in dataloaders_list:
        with torch.no_grad():
            for batch_idx , (data, target) in enumerate(dataloader):
                if batches is not None and batch_idx >= batches:
                    break
                for (model_dir, model) in model_list:
                    model.eval()
                    data, target = data.to(device), target.to(device)

                    orig_out = model(data)

                    orig_conf, orig_pred = torch.max(torch.softmax(orig_out, dim=1), 1)

                    for a_i, (attack_name, attack) in enumerate(attacks):
                        attack.set_model(model)

                        #cpu
                        data_targeted_attacked = torch.zeros((num_classes, ) + data.shape)
                        pred_targeted_attacked = torch.zeros((num_classes, data.shape[0]), dtype=torch.int32)
                        conf_targeted_attacked = torch.zeros((num_classes, data.shape[0]))

                        #creates batches that have dimension batch_size * num_classes
                        stacked_data = data.repeat(num_classes, 1, 1, 1)
                        stacked_targets = target.new_empty( target.shape[0] * num_classes, dtype=torch.long)
                        for c in range(num_classes):
                            stacked_targets[(c * num_classes):((c+1)*num_classes)] = c

                        data_perturbed = attack.perturb(stacked_data, stacked_targets, True)
                        out = model(data_perturbed)
                        conf, pred = torch.max(torch.softmax(out, dim=1), 1)

                        #might be possible to reshape this?
                        for c in range(num_classes):
                            conf_targeted_attacked[c, :] = conf[(c * num_classes):((c+1)*num_classes)]
                            pred_targeted_attacked[c, :] = pred[(c * num_classes):((c+1)*num_classes)]
                            data_targeted_attacked[c, :] = data_perturbed[(c * num_classes):((c+1)*num_classes)].to('cpu')

                        #print with
                        out_folder = f'{base_dir}/{model_dir}/{dataloader_name}'
                        out_file = f'{out_folder}/{attack_name}_{batch_idx}.png'
                        pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

                        fig, ax = plt.subplots( data.shape[0], num_classes + 1, figsize=(20,22))

                        for i in range( data.shape[0] ):
                            #plot original
                            img_orig = data[i, :].detach().cpu().permute(1, 2, 0)
                            if img_orig.shape[2] == 1:
                                img_orig.squeeze_()
                                ax[i, 0].imshow(img_orig, cmap='gray')
                            else:
                                ax[i, 0].imshow(img_orig)

                            ax[i, 0].axis('off')
                            ax[i, 0].title.set_text('{:d} - {:0.2f}'.format(orig_pred[i].item(),
                                                                            orig_conf[i].item()))

                            for j in range( num_classes):
                                img_ij = data_targeted_attacked[j, i, :].permute(1, 2, 0)
                                if img_ij.shape[2] == 1:
                                    img_ij.squeeze_()
                                    ax[i, j+1].imshow(img_ij, cmap='gray')
                                else:
                                    ax[i, j+1].imshow(img_ij)

                                ax[i,j + 1].axis('off')
                                ax[i,j + 1].title.set_text('{:d} - {:0.2f}'.format(pred_targeted_attacked[j,i].item(), conf_targeted_attacked[j,i].item()))

                        fig.savefig(out_file)
                        plt.close(fig)

def test_robustness(model, attacks, device, test_loader):
    model.eval()

    test_loss = np.zeros(len(attacks))
    correct = np.zeros(len(attacks))
    av_conf = np.zeros(len(attacks))

    for attack in attacks:
        attack.set_model(model)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            for a_i, attack in enumerate(attacks):
                data_perturbed = attack.perturb(data, target)
                output = model(data_perturbed)

                loss = F.cross_entropy(output, target)
                test_loss[a_i] += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss

                c, pred = output.max(1, keepdim=True)  # get the index of the max log-probability
                correct[a_i] += (pred.eq(target.view_as(pred))).sum().item()
                av_conf[a_i] += c.exp().sum().item()
                test_loss += loss.item()


    test_loss /= len(test_loader.dataset)
    av_conf /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)

    return correct, av_conf, test_loss

