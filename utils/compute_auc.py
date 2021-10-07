import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.train_types.train_loss import MaxConfidenceLoss, NegativeWrapper
from utils.adversarial_attacks import APGDAttack, ArgminPGD
import numpy as np
def fpr_at_tpr(values_in, values_out, tpr):
    in_np = values_in.detach().cpu().numpy()
    out_np = values_out.detach().cpu().numpy()
    t = np.quantile(in_np, (1-tpr))
    fpr = (out_np >= t).mean()
    return fpr

def auprc(values_in, values_out):
    y_true = len(values_in)*[1] + len(values_out)*[0]
    y_score = np.concatenate([values_in, values_out])
    return sklearn.metrics.average_precision_score(y_true, y_score)



def _get_conf(model, device, test_loader, max_samples=1e13):
    conf = []
    mean_conf = 0
    correct = 0

    samples_collected = 0
    with torch.no_grad():
        for data, target in test_loader:
            if samples_collected >= max_samples:
                break

            samples_left = min(data.shape[0], max_samples - samples_collected)
            data = data[:samples_left,:]
            target = target[:samples_left]

            data, target = data.to(device), target.to(device)
            out = model(data)

            if out.dim() > 1 and out.shape[1] > 1:
                c, pred = F.softmax(out, dim=1).max(dim=1)
            else:
                pred = torch.zeros_like(out,dtype=torch.long)
                c = torch.sigmoid(out.squeeze())

            correct += torch.sum(pred.eq(target)).item()
            mean_conf += torch.sum(c).item()
            conf.append(c.cpu())

            samples_collected += c.shape[0]

    conf = torch.cat(conf, 0)
    acc = correct / samples_collected
    mean_conf = mean_conf / samples_collected
    return conf, acc, mean_conf

def _get_wc_conf(model, device, test_loader, eps, num_classes, max_samples=1e13):
    conf = []
    mean_conf = 0
    correct = 0

    loss = NegativeWrapper(MaxConfidenceLoss())
    att = APGDAttack(model, num_classes, eps=eps, n_iter=100, norm='L2', n_restarts=1, loss=loss, eot_iter=1)

    samples_collected = 0

    with torch.no_grad():
        for data, target in test_loader:
            if samples_collected >= max_samples:
                break

            samples_left = min(data.shape[0], max_samples - samples_collected)
            data = data[:samples_left, :]
            target = target[:samples_left]

            data, target = data.to(device), target.to(device)
            adv_samples = att(data, target)

            out = model(adv_samples)

            c, pred = F.softmax(out, dim=1).max(dim=1)

            correct += torch.sum(pred.eq(target)).item()
            mean_conf += torch.sum(c).item()
            conf.append(c.cpu())

            samples_collected += c.shape[0]

    conf = torch.cat(conf, 0)
    acc = correct / samples_collected
    mean_conf = mean_conf / samples_collected
    return conf, acc, mean_conf

def _get_auroc(conf_in, conf_out):
    y_true = torch.cat([torch.ones_like(conf_in.cpu()),
                        torch.zeros_like(conf_out)]).cpu().numpy()
    y_scores = torch.cat([conf_in.cpu(),
                          conf_out]).cpu().numpy()
    success_rate = (conf_out >= conf_in.median()).float().mean().item()
    auroc = roc_auc_score(y_true, y_scores)
    return auroc, success_rate

def compute_auc(model, in_loader, out_loaders, device, auc_samples=1e13):
    od_mmcs = torch.zeros(len(out_loaders))
    aucs = torch.zeros_like(od_mmcs)
    fpr95 = torch.zeros_like(od_mmcs)

    conf_in, acc_in, mean_conf = _get_conf(model, device, in_loader, max_samples=auc_samples)
    print(f'ID Accuracy {acc_in} - ID MMC {mean_conf}')

    for loader_idx, (dataset_name, loader) in enumerate(out_loaders):
        eps_conf_out, _, od_mean_conf = _get_conf(model, device, loader, max_samples=auc_samples)

        auc = _get_auroc(conf_in, eps_conf_out)[0]
        fpr = fpr_at_tpr(conf_in, eps_conf_out, 0.95)

        aucs[loader_idx] = auc
        od_mmcs[loader_idx] = od_mean_conf
        fpr95[loader_idx] = fpr
        print(f'AUC {dataset_name} - {auc} - FPR95 {fpr} - MMC {od_mean_conf}')

    eps_auc_average = torch.mean(aucs, dim=0)
    eps_mmc_avergage = torch.mean(od_mmcs, dim=0)
    eps_fpr_avergage = torch.mean(fpr95, dim=0)

    print(f'AUC average {eps_auc_average}')
    print(f'MMC average {eps_mmc_avergage}')
    print(f'FPR average {eps_fpr_avergage}')
    return aucs

def compute_wc_auc(model, in_loader, out_loaders, device, num_classes, auc_samples, auc_eps):
    od_mmcs = torch.zeros((len(out_loaders), len(auc_eps)))
    aucs = torch.zeros((len(out_loaders), len(auc_eps)))
    fpr95 = torch.zeros((len(out_loaders), len(auc_eps)))

    for eps_idx, eps in enumerate(auc_eps):
        conf_in, acc_in, mean_conf = _get_conf(model, device, in_loader, max_samples=auc_samples[eps_idx])
        print(f'ID Accuracy {acc_in} - ID MMC {mean_conf}')

        for loader_idx, (dataset_name, loader) in enumerate(out_loaders):
            if eps > 0:
                eps_conf_out, _, od_mean_conf = _get_wc_conf(model, device, loader, eps, num_classes, max_samples=auc_samples[eps_idx])
            else:
                eps_conf_out, _, od_mean_conf = _get_conf(model, device, loader, max_samples=auc_samples[eps_idx])

            eps_auc = _get_auroc(conf_in, eps_conf_out)[0]
            eps_fpr = fpr_at_tpr(conf_in, eps_conf_out, 0.95)

            aucs[loader_idx, eps_idx] = eps_auc
            od_mmcs[loader_idx, eps_idx] = od_mean_conf
            fpr95[loader_idx, eps_idx] =  eps_fpr
            print(f'WorstCase AUC {dataset_name} - {eps} - {eps_auc} - FPR95 {eps_fpr} - MMC {od_mean_conf}')

    eps_auc_average = torch.mean(aucs, dim=0)
    eps_mmc_avergage = torch.mean(od_mmcs, dim=0)
    eps_fpr_avergage = torch.mean(fpr95, dim=0)
    print(f'AUC average {eps_auc_average}')
    print(f'MMC average {eps_mmc_avergage}')
    print(f'FPR average {eps_fpr_avergage}')

    return aucs
