import numpy as np
import pytest

from cached_interpolate.interpolate import RegularCachingInterpolant


class _Foo:
    """Dummy class to mimic how this is used in GWPopulation"""

    def __init__(self, x, y, kind, backend, interpolant):
        from functools import partial

        self.interpolant = partial(
            interpolant(x=x, y=x, kind=kind, backend=backend),
            y,
        )

    def __call__(self, data):
        self.interpolant(data)


def test_caching_with_jax(kind, interpolant):
    """
    Create the interpolant and run a few times with various inputs and
    compilation to test
    https://github.com/ColmTalbot/cached_interpolation/issues/19
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    test_values = np.random.uniform(0, 1, (2, 10000))
    kwargs = dict(
        x=np.linspace(0, 1, 5),
        y=test_values,
        kind=kind,
        backend=jnp,
        interpolant=interpolant,
    )
    spl = _Foo(**kwargs)

    test_values = np.asarray(np.random.uniform(0, 1, 5))
    _ = spl(test_values)
    test_values = jnp.asarray(test_values)
    temp = jax.jit(spl)
    _ = temp(jnp.asarray(np.random.uniform(0, 1, 5)))
    _ = temp(test_values)


def test_running_without_new_y_values():
    jax = pytest.importorskip("jax")
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = RegularCachingInterpolant(x_values, y_values, kind="cubic", backend=jax.numpy)

    @jax.jit
    def func(xvals, yvals):
        return spl(xvals, yvals, use_cache=False)

    old_values = spl._data
    vals1 = func(x_values, y_values)
    points = jax.numpy.asarray(np.random.uniform(-1, 1, 10))
    vals2 = func(np.array([0, 1]), points)

    assert vals1.shape != vals2.shape
    # assert np.max(old_values - spl._data) > 1e-5
