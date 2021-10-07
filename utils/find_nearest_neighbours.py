import torch
import torch.nn.functional as F
import pathlib
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import lpips

class L2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ref_point, batch):
        batch = batch.view(batch.shape[0], -1)
        ref_point = ref_point.view(1, -1)

        l2 = torch.sqrt(torch.sum(((batch - ref_point) ** 2), dim=1))
        return l2

class FeatureDist(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ref_point, batch):
        all_data = torch.cat([ref_point, batch])
        out = self.model(all_data)
        ref_feature = out[0,:].view(1, -1)
        batch_features = out[1:, :].view(batch.shape[0], -1)
        l2 = torch.sqrt(torch.sum(((batch_features - ref_feature) ** 2), dim=1))
        return l2


class LPIPS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net='alex')

    def forward(self, ref_point, batch):
        ref_batch = ref_point.expand_as(batch)
        sim = self.loss_fn(ref_batch, batch).squeeze()
        return sim

def find_nearest_neighbours(dist_function, ref_batch, data_loader, data_set, device, num_neighbours, out_dir, out_prefix, is_similarity=False):
    distances = torch.zeros(ref_batch.shape[0], len(data_set))
    ref_batch = ref_batch.to(device)

    data_idx = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)

            for ref_i in range(ref_batch.shape[0]):
                ref_point = ref_batch[ref_i].unsqueeze(0)
                d_data = dist_function(ref_point, data)

                distances[ref_i, data_idx:(data_idx+data.shape[0])] = d_data.detach().cpu()

            data_idx += data.shape[0]

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ref_i in range(ref_batch.shape[0]):

        num_cols = 1 + num_neighbours
        scale_factor = 2
        fig, ax = plt.subplots(1, num_cols, figsize=(scale_factor * num_cols, 1.3 * scale_factor))
        ax = np.expand_dims(ax, axis=0)
        # plot original:
        ax[0, 0].axis('off')
        ax[0, 0].title.set_text(f'Target')
        target_img = ref_batch[ref_i, :].permute(1, 2, 0).cpu().detach()
        ax[0, 0].imshow(target_img, interpolation='lanczos')

        d_i = distances[ref_i, :]
        if is_similarity:
            d_i_sort_idcs = torch.argsort(d_i, descending=True)
        else:
            d_i_sort_idcs = torch.argsort(d_i, descending=False)

        for j in range(num_neighbours):
            j_idx = d_i_sort_idcs[j]
            ref_img = data_set[j_idx][0].permute(1, 2, 0).cpu().detach()
            ax[0, j + 1].axis('off')
            ax[0, j + 1].imshow(ref_img, interpolation='lanczos')
            d_j = d_i[j_idx]
            ax[0, j + 1].title.set_text(f'{d_j:.3f}')

        plt.tight_layout()

        fig.savefig(os.path.join(out_dir, f'{out_prefix}_{ref_i}.png'))
        fig.savefig(os.path.join(out_dir, f'{out_prefix}_{ref_i}.pdf'))
        plt.close(fig)
