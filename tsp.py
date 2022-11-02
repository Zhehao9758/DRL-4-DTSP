import matplotlib
import numpy as np
import torch
from torch.utils.data import Dataset

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TSPDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=None):
        super(TSPDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)

        original_location = torch.rand((int(num_samples), 2, size))
        original_mask = torch.zeros(int(num_samples), 1, size)
        self.static = original_location
        self.dynamic = torch.cat((original_mask, original_location), 1)

        self.num_nodes = size
        self.size = num_samples
        self.mean = 0.0
        self.std = 0.02

    def __len__(self):
        return self.size

    def __getitem__(self, idx):  # 取一个样本
        return self.static[idx], self.dynamic[idx], []

    def update_dynamic(self, dynamic, chosen_idx): # 把访问的dynamic变成1，变化坐标

        visit = chosen_idx.ne(0)
        dynamic_copy = dynamic.clone()

        if visit.any():

            for i in range(len(dynamic_copy)):
                dynamic_copy[i][0][chosen_idx] = 1

            for i in range(len(dynamic_copy)):
                non_visit_set = [j for j in range(self.num_nodes) if dynamic_copy[i][0][j] == 0]
                for idx in non_visit_set:
                    dynamic_copy[i][1][idx] += torch.normal(self.mean, self.std, (1, 1)).squeeze()
                    dynamic_copy[i][2][idx] += torch.normal(self.mean, self.std, (1, 1)).squeeze()
        return dynamic_copy.clone().detach()


def update_mask(mask,dynamic ,chosen_idx):  # 变0就不可选了

    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask

def reward(dynamic, tour_indices):

    final_coordinates = dynamic[:, 1:, :]
    idx = tour_indices.unsqueeze(1).expand_as(final_coordinates)
    tour = torch.gather(final_coordinates.data, 2, idx).permute(0, 2, 1)
    y = torch.cat((tour, tour[:, :1]), dim=1)
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1).detach()








def render(dynamic, tour_indices, save_path):

    static = dynamic[:, 1:, :]
    plt.close('all')
    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1
    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')
    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)
        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)


def render_static(static, tour_indices, save_path):

    plt.close('all')
    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):


        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)


        idx = idx.expand(static.size(1), -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)

        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
