import math
from torch.nn.functional import softplus
from torch.distributions import constraints, TransformedDistribution, Normal
from torch.distributions.transforms import Transform, AffineTransform


class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))


class TanhTransformedGaussian(TransformedDistribution):
    def __init__(self, mu, scale):
        self.mu, self.scale = mu, scale
        base_dist = Normal(mu, scale)
        transforms = [TanhTransform(cache_size=1)]
        super(TanhTransformedGaussian, self).__init__(base_dist, transforms)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return None

    def enumerate_support(self, expand=True):
        pass

    def entropy(self):
        pass


class ScaledTanhTransformedGaussian(TransformedDistribution):
    def __init__(self, mu, scale, min=-1.0, max=1.0):
        self.mu, self.scale = mu, scale
        base_dist = Normal(mu, scale)
        transforms = [TanhTransform(cache_size=1), AffineTransform(loc=0, scale=(max - min)/2)]
        super().__init__(base_dist, transforms)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return None

    def enumerate_support(self, expand=True):
        pass

    def entropy(self):
        pass