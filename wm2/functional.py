from math import pi

import numpy as np
import torch


def squared_diff(h, len):
    """
    keypoints, K
    :param h: K, 1
    :param len: integer
    :return: heightmap
    """
    ls = np.linspace(0, 1, len, dtype=h.dtype)
    ls = np.expand_dims(ls, axis=0).repeat(repeats=h.shape[0], axis=0)
    hm = np.expand_dims(h, axis=0).repeat(repeats=len, axis=0).T
    hm = ls - hm
    return hm ** 2


def gaussian_like_function(kp, height, width, sigma=0.1, eps=1e-6):
    hm = squared_diff(kp[:, 0], height)
    wm = squared_diff(kp[:, 1], width)
    hm = hm[:, :, np.newaxis]
    hm = np.repeat(hm, repeats=width, axis=2)
    wm = wm[:, np.newaxis, :]
    wm = np.repeat(wm, repeats=height, axis=1)
    gm = - np.sqrt(hm + wm + eps) / (2 * sigma ** 2)
    gm = np.exp(gm)
    return gm


def multivariate_gaussian(mu, covar, size):
    """

    :param mu: mean (N, D)
    :param covar: std-deviation (not variance!) (N, D, D)
    :param size: tuple of output dimension sizes eg: (h, w)
    :return: a heightmap
    """
    N = mu.size(0)
    D = mu.size(1)
    axes = [torch.linspace(0, 1.0, size[i], dtype=mu.dtype, device=mu.device) for i in range(D)]
    grid = torch.meshgrid(axes)
    grid = torch.stack(grid)
    gridshape = grid.shape[1:]
    grid = grid.flatten(start_dim=1)
    eps = torch.finfo(mu.dtype).eps

    constant = torch.tensor((2 * pi) ** D).sqrt() * torch.det(covar)
    mean_diff = (grid.view(1, D, -1) - mu.view(N, D, 1))
    covar = torch.cholesky(covar + eps)
    inv_covar = torch.cholesky_inverse(covar)
    exponent = torch.einsum('bij, jk, bki -> bi', mean_diff.permute(0, 2, 1), inv_covar, mean_diff) * - 0.5
    #exponent2 = mean_diff.permute(0, 2, 1).matmul(inv_covar).matmul(mean_diff) * - 0.5
    #exponent2 = exponent2.diagonal(dim1=1, dim2=2)

    exponent = torch.exp(exponent)
    height = exponent / constant
    return height.reshape(N, *gridshape)


def multivariate_diag_gaussian(mu, stdev, size):
    """

    :param mu: mean (N, D)
    :param stdev: std-deviation (not variance!) (N, D)
    :param size: tuple of output dimension sizes eg: (h, w)
    :return: a heightmap
    """
    N = mu.size(0)
    D = mu.size(1)
    axes = [torch.linspace(0, 1.0, size[i], dtype=mu.dtype, device=mu.device) for i in range(D)]
    grid = torch.meshgrid(axes)
    grid = torch.stack(grid)
    gridshape = grid.shape[1:]
    grid = grid.flatten(start_dim=1)
    eps = torch.finfo(mu.dtype).eps
    stdev[stdev < eps] = stdev[stdev < eps] + eps

    constant = torch.tensor((2 * pi) ** D).sqrt() * torch.prod(stdev, dim=1)
    numerator = (grid.view(1, D, -1) - mu.view(N, D, 1)) ** 2
    denominator = 2 * stdev ** 2
    exponent = - torch.sum(numerator / denominator.view(N, D, 1), dim=1)
    exponent = torch.exp(exponent)
    height = exponent / constant
    return height.reshape(N, *gridshape)


def compute_covar(samples, mu):
    covar = samples - mu.view(1, *mu.shape)
    s, n, t, d = samples.shape
    covar = covar.reshape(s, n * t, d)
    covar = covar.permute(1, 2, 0).matmul(covar.permute(1, 0, 2)) / (s - 1)
    covar = covar.reshape(n, t, d, d)
    return covar


def simple_sample(n, mean, c):
    z = torch.randn(*mean.shape, n, device=mean.device, dtype=mean.dtype)
    s = mean.view(*mean.shape, 1) + c.matmul(z)
    return s.permute(0, 1, 3, 2)