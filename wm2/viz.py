from time import sleep

from math import floor

import cv2
import numpy as np
import torch

from functional import multivariate_diag_gaussian, multivariate_gaussian


def to_opencv_image(im):
    '''Convert to OpenCV image shape h,w,c'''
    shape = im.shape
    if len(shape) == 3 and shape[0] < shape[-1]:
        return im.transpose(1, 2, 0)
    else:
        return im


def debug_image(im, block=True):
    '''
    Renders an image for debugging; pauses process until key press
    Handles tensor/numpy and conventions among libraries
    '''
    if torch.is_tensor(im):  # if PyTorch tensor, get numpy
        im = im.cpu().numpy()
    im = to_opencv_image(im)
    im = im.astype(np.uint8)  # typecast guard
    if im.shape[0] == 3:  # RGB image
        # accommodate from RGB (numpy) to BGR (cv2)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow('debug image', im)
    if block:
        cv2.waitKey(0)
    else:
        cv2.waitKey(10)


def strip_index(center, thickness, length):
    center = floor(center * length)
    thickness = floor(thickness * length // 2)
    lower, upper = center - thickness, center + thickness
    return lower, upper


def put_strip(image_size, center, thickness, dim, mu, stdev=None, covar=None):
    image = np.zeros(image_size)
    # stdev = torch.tensor([[0.05]], device=device)
    if stdev is not None:
        strip = multivariate_diag_gaussian(mu.view(1, -1), stdev.view(1, -1), image_size)
    elif covar is not None:
        strip = multivariate_gaussian(mu.view(1, -1), covar, image_size)
    else:
        raise Exception('required either a stdev or covariance matrix')

    strip = strip.cpu().numpy()
    lower, upper = strip_index(center, thickness, image_size[dim])
    if dim == 1:
        image[:, lower:upper] = strip.T
    elif dim == 0:
        image[lower:upper, :] = strip
    image = ((image / np.max(image)) * 255).astype(np.uint)
    return image


def put_gaussian(image_size, mu, stdev=None, covar=None):
    # stdev = torch.tensor([[0.05, 0.05]], device=device)
    if stdev is not None:
        point = multivariate_diag_gaussian(mu.view(1, -1), stdev.view(1, -1), image_size)
    elif covar is not None:
        point = multivariate_gaussian(mu.view(1, -1), covar, image_size)
    else:
        raise Exception('required either a stdev or covariance matrix')
    image = point.cpu().numpy()
    image = ((image / np.max(image)) * 255).astype(np.uint)
    return image


def display_predictions(trajectory_mu, trajectory_stdev=None, trajectory_covar=None, label=None, max_length=12 * 5,
                        fps=12, scale=4, max_lookahead=3):

        length = min(trajectory_mu.size(1), max_length)
        image_size = (240 * scale, 160 * scale)

        for step in range(length):
            mu = trajectory_mu[:, step]
            covar = trajectory_covar[:, step]
            panel = []
            for t in range(max_lookahead):
                if label == 'all':
                    player = put_strip(image_size, 0.9, 0.05, dim=1, mu=mu[0], covar=covar[0])
                    enemy = put_strip(image_size, 0.3, 0.05, dim=1, mu=mu[1], covar=covar[1])
                    ball = put_gaussian(image_size, mu[2:4], covar[2:4])
                    image = np.stack((player, enemy, ball.squeeze()))

                elif label == 'player':
                    image = put_strip(image_size, 0.9, 0.05, dim=1, mu=mu[t], covar=covar[t])

                elif label == 'enemy':
                    image = put_strip(image_size, 0.9, 0.05, dim=1, mu=mu[t], covar=covar[t])

                elif label == 'ball':
                    image = put_gaussian(image_size, mu[t], covar=covar[t]).squeeze()
                else:
                    raise Exception(f'label {label} not found')

                panel.append(image)

            image = np.concatenate(panel, axis=1)
            debug_image(image, block=False)
            sleep(1 / fps)