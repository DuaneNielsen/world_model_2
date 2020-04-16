from matplotlib import pyplot as plt
from wm2.distributions import TanhTransformedGaussian, ScaledTanhTransformedGaussian
import torch

def test_tanh_transform():
    dist = TanhTransformedGaussian(1.0, 0.2)
    samples = dist.sample(sample_shape=(1000,))
    plt.hist(samples.numpy(), bins=50)
    plt.show()

def test_scaled_tanh_transform():
    dist = ScaledTanhTransformedGaussian(6.0, 0.2, min=-2.0, max=+2.0)
    samples = dist.sample(sample_shape=(1000,))
    plt.hist(samples.numpy(), bins=50)
    plt.show()


def test_tanh():
    x = torch.linspace(-1.0, 1.0, 20)
    y = x.tanh()
    plt.plot(x, y)
    plt.show()