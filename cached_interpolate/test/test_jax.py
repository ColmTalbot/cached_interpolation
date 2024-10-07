import numpy as np
import pytest
from cached_interpolate.interpolate import RegularCachingInterpolant


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
