from matplotlib import pyplot as plt
from wm2.distributions import TanhTransformedGaussian
import torch

def test_tanh_transform():
    dist = TanhTransformedGaussian(0.0, 0.8)
    samples = dist.sample(sample_shape=(1000,))
    plt.hist(samples.numpy(), bins=50)
    plt.show()

def test_tanh():
    x = torch.linspace(-1.0, 1.0, 20)
    y = x.tanh()
    plt.plot(x, y)
    plt.show()