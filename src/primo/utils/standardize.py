from __future__ import annotations

import random

import numpy as np
import torch


class Standardize:
    """
    standardize latent space
    """

    def __init__(self, data_all, model, device):

        self.onehot_data = self.random_select(data_all)
        self.model = model
        self.device = device
        self.estimate_stand()

    def estimate_stand(self):
        batch_size = 2500

        Z_sample = np.zeros((len(self.onehot_data), 196))

        for chunk in self.chunks(list(range(len(self.onehot_data))), batch_size):
            one_hot = torch.tensor(self.onehot_data[chunk]).float().to(self.device)
            _, _, _, embdata_torch = self.model(one_hot)
            Z_sample[chunk, :] = embdata_torch.cpu().detach().numpy()
        self.mu = np.nanmean(Z_sample, axis=0)
        self.std = np.std(Z_sample, axis=0)
        return

    def standardize_z(self, z):
        return (z - self.mu) / self.std

    def standardize_z_torch(self, z):
        return (z - torch.tensor(self.mu).float().to(self.device)) / torch.tensor(
            self.std
        ).float().to(self.device)

    def random_select(self, data, size=50000):
        indices = random.sample(range(len(data)), size)
        return data[indices]

    @staticmethod
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i : (i + n)]


class StandardizeLoad:
    """
    standardize latent space with known global means and stds
    """

    def __init__(self, mu, std, device):
        self.mu = mu
        self.std = std
        self.device = device

    def standardize_z(self, z):
        return (z - self.mu) / self.std

    def standardize_z_torch(self, z):
        return (z - torch.tensor(self.mu).float().to(self.device)) / torch.tensor(
            self.std
        ).float().to(self.device)
