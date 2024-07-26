import numpy as np
import pytest
from scipy.interpolate import CubicSpline, interp1d

from cached_interpolate import RegularCachingInterpolant


@pytest.mark.parametrize("bc_type", ["clamped", "natural", "not-a-knot", "periodic"])
def test_cubic_matches_scipy(bc_type, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    if bc_type == "periodic":
        y_values[0] = y_values[-1]
    spl = RegularCachingInterpolant(
        x_values, y_values, kind="cubic", bc_type=bc_type, backend=backend
    )
    test_points = np.random.uniform(0, 1, 10000)
    max_diff = 0
    for _ in range(100):
        y_values = np.random.uniform(-1, 1, 10)
        if bc_type == "periodic":
            y_values[0] = y_values[-1]
        scs = CubicSpline(x=x_values, y=y_values, bc_type=bc_type)
        diffs = spl(test_points, y=y_values) - scs(test_points)
        max_diff = max(np.max(diffs), max_diff)
    assert max_diff, 1e-10


def test_caching_interpolant_bad_bc_type(interpolant):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    with pytest.raises(NotImplementedError):
        _ = interpolant(x_values, y_values, kind="cubic", bc_type="bad")


def test_nearest_matches_scipy(interpolant, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = interpolant(x_values, y_values, kind="nearest", backend=backend)
    test_points = np.random.uniform(0, 1, 10000)
    max_diff = 0
    for _ in range(100):
        y_values = np.random.uniform(-1, 1, 10)
        scs = interp1d(x=x_values, y=y_values, kind="nearest")
        diffs = spl(test_points, y=y_values) - scs(test_points)
        max_diff = max(np.max(diffs), max_diff)
    assert max_diff < 1e-10


def test_linear_matches_numpy(interpolant, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = interpolant(x_values, y_values, kind="linear", backend=backend)
    test_points = np.random.uniform(0, 1, 10000)
    max_diff = 0
    for _ in range(100):
        y_values = np.random.uniform(-1, 1, 10)
        npy = np.interp(test_points, x_values, y_values)
        diffs = spl(test_points, y=y_values) - npy
        max_diff = max(np.max(diffs), max_diff)
    assert max_diff < 1e-10


def test_single_input(interpolant, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = interpolant(x_values, y_values, kind="cubic", backend=backend)
    assert spl(0) == y_values[0]


def test_single_complex(interpolant, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    y_values = y_values + 1j * (1 - y_values)
    spl = interpolant(x_values, y_values, kind="cubic", backend=backend)
    assert spl(0) == y_values[0]


def test_interpolation_at_lower_bound(interpolant, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = interpolant(x_values, y_values, kind="cubic", backend=backend)
    test_point = 0
    assert abs(spl(test_point) - y_values[0]) < 1e-5


def test_interpolation_at_upper_bound(interpolant, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = interpolant(x_values, y_values, kind="cubic", backend=backend)
    test_point = 1
    assert abs(spl(test_point) - y_values[-1]) < 1e-5


def test_bad_interpolation_method_raises_error(interpolant):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    with pytest.raises(ValueError):
        _ = interpolant(x_values, y_values, kind="bad method")


def test_running_without_new_y_values(interpolant, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = interpolant(x_values, y_values, kind="cubic", backend=backend)
    old_values = spl._data
    points = backend.asarray(np.random.uniform(-1, 1, 10))
    _ = spl(np.array([0, 1]), y=points, use_cache=False)
    assert np.max(old_values - spl._data) > 1e-5


def test_running_with_complex_input_linear(interpolant, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    y_values = y_values * np.exp(1j * np.random.uniform(0, 2 * np.pi, 10))
    spl = interpolant(x_values, y_values, kind="linear", backend=backend)
    scs = interp1d(x=x_values, y=y_values, kind="linear")
    test_points = backend.asarray(np.random.uniform(0, 1, 10))
    scs_test = scs(test_points)
    diffs = spl(test_points) - scs_test
    assert np.max(diffs) < 1e-10


def test_running_with_complex_input_cubic(interpolant, backend):
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    y_values = y_values * np.exp(1j * np.random.uniform(0, 2 * np.pi, 10))
    spl = interpolant(
        x_values, y_values, kind="cubic", bc_type="natural", backend=backend
    )
    scs = CubicSpline(x=x_values, y=y_values, bc_type="natural")
    test_points = backend.asarray(np.random.uniform(0, 1, 10))
    scs_test = scs(test_points)
    diffs = spl(test_points) - scs_test
    assert np.max(diffs) < 1e-10


def test_2d_input(kind, interpolant, backend):
    kwargs = dict(
        x=np.linspace(0, 1, 5),
        y=np.random.uniform(0, 1, 5),
        kind=kind,
        backend=backend,
    )
    test_values = backend.asarray(np.random.uniform(0, 1, (2, 10000)))
    spl = interpolant(**kwargs)
    array_test = spl(test_values)
    loop_test = list()
    for ii in range(2):
        spl = interpolant(**kwargs)
        loop_test.append(spl(test_values[ii]))
    loop_test = np.array(loop_test)
    assert np.array_equal(loop_test, array_test)


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
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax import jit

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
    temp = jit(spl)
    _ = temp(jnp.asarray(np.random.uniform(0, 1, 5)))
    _ = temp(test_values)
