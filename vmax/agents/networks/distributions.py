import abc
import jax
import jax.numpy as jnp

__copyright__ = "Copyright 2024 The Brax Authors."
__license__ = "Apache-2.0"
__modifications__ = "Adapted for TRUST's normal and tanh action distribution components."


class ParametricDistribution(abc.ABC):

    def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
        self._param_size = param_size
        self._postprocessor = postprocessor
        self._event_ndims = event_ndims
        self._reparametrizable = reparametrizable
        assert event_ndims in [0, 1]

    @abc.abstractmethod
    def create_dist(self, parameters):
        pass

    @property
    def param_size(self):
        return self._param_size

    @property
    def reparametrizable(self):
        return self._reparametrizable

    def postprocess(self, event):
        return self._postprocessor.forward(event)

    def inverse_postprocess(self, event):
        return self._postprocessor.inverse(event)

    def sample_no_postprocessing(self, parameters, seed):
        return self.create_dist(parameters).sample(seed=seed)

    def sample(self, parameters, seed):
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    def mode(self, parameters):
        return self.postprocess(self.create_dist(parameters).mode())

    def log_prob(self, parameters, actions):
        dist = self.create_dist(parameters)
        log_probs = dist.log_prob(actions)
        log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
        if self._event_ndims == 1:
            log_probs = jnp.sum(log_probs, axis=-1)
        return log_probs


class NormalDistribution:

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, seed):
        return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

    def mode(self):
        return self.loc

    def log_prob(self, x):
        log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        return log_unnormalized - log_normalization


class TanhBijector:

    def forward(self, x):
        return jnp.tanh(x)

    def inverse(self, y):
        return jnp.arctanh(y)

    def forward_log_det_jacobian(self, x):
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))


class NormalTanhDistribution(ParametricDistribution):

    def __init__(self, event_size, min_std=0.001):
        super().__init__(
            param_size=2 * event_size,
            postprocessor=TanhBijector(),
            event_ndims=1,
            reparametrizable=True,
        )
        self._min_std = min_std

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = jax.nn.softplus(scale) + self._min_std
        return NormalDistribution(loc=loc, scale=scale)
