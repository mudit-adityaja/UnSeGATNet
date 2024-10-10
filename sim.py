import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from torchvision.transforms import RandomAffine

import numpy as np
import os

class TensorAugmentation:
    def __init__(self, noise_level=0.05, dropout_prob=0.2, scaling_factor_range=(0.8, 1.2), shifting_range=(-0.1, 0.1)):
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob
        self.scaling_factor_range = scaling_factor_range
        self.shifting_range = shifting_range

    def add_noise(self, tensor):
        if self.noise_level > 0:
            noise = torch.randn(tensor.size()) * self.noise_level
            tensor = tensor + noise.to('cuda')
        return tensor

    def feature_dropout(self, tensor):
        mask = (torch.rand(tensor.size()) > self.dropout_prob).float()
        return tensor * mask.to('cuda')

    def scale(self, tensor):
        scaling_factor = np.random.uniform(*self.scaling_factor_range)
        return tensor * torch.tensor(scaling_factor).to('cuda')

    def shift(self, tensor):
        shift_value = np.random.uniform(*self.shifting_range)
        return tensor + torch.tensor(shift_value).to('cuda')

    def apply_transformations(self, tensor):
        # Randomly choose transformations to apply
        if np.random.rand() < 0.5:
            tensor = self.add_noise(tensor)
        if np.random.rand() < 0.5:
            tensor = self.feature_dropout(tensor)
        if np.random.rand() < 0.5:
            tensor = self.scale(tensor)
        if np.random.rand() < 0.5:
            tensor = self.shift(tensor)

        return tensor


def D(p, z):
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

# def random_affine_transform(x):
#     transform = RandomAffine(degrees=(0, 360), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(0, 10))
#     transformed = torch.zeros_like(x)
#     for i in range(x.size(0)):
#         transformed[i] = transform(x[i].unsqueeze(0)).squeeze(0)
#     return transformed

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Projector, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            # nn.BatchNorm1d(input_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.fc(x)

def get_wsa(F, bs=64, epochs=10):
    x, y = F.shape
    F = torch.from_numpy(F).float()
    dataset = TensorDataset(F)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=os.cpu_count())
    projector = Projector(y, 256).to('cuda') # y = 384
    predictor = Predictor(y, 256).to('cuda')
    optimizer = optim.Adam(list(projector.parameters()) + list(predictor.parameters()), lr=0.001)
    augmentor = TensorAugmentation()

    for epoch in range(epochs):
        for batch in dataloader:
            phi_I = batch[0].to('cuda')
            optimizer.zero_grad()

            # alpha = random_affine_transform(phi_I)
            # beta = random_affine_transform(phi_I)
            alpha = augmentor.apply_transformations(phi_I)
            beta = augmentor.apply_transformations(phi_I)

            delta_alpha = projector(alpha)
            delta_beta = projector(beta)
            f_alpha = predictor(delta_alpha)
            f_beta = predictor(delta_beta)

            loss = 0.5 * (D(f_alpha, delta_beta.detach()) + D(delta_alpha.detach(), f_beta))
            loss.backward()
            optimizer.step()

    all_alpha_outputs = None
    for batch in dataloader:
        phi_I = batch[0].to('cuda')
        # alpha = random_affine_transform(phi_I)
        alpha = augmentor.apply_transformations(phi_I)
        out = predictor(alpha)
        if all_alpha_outputs is not None:
          all_alpha_outputs = torch.cat((all_alpha_outputs, out), dim=0)
        else:
          all_alpha_outputs = out

    # all_alpha_outputs = torch.cat(all_alpha_outputs, dim=0)

    all_alpha_outputs = all_alpha_outputs.detach().cpu().numpy()
    out_norm = all_alpha_outputs / np.linalg.norm(all_alpha_outputs, axis=1, keepdims=True)
    Wsa = np.dot(out_norm, out_norm.T)
    return Wsa






