import sys

print(sys.path)  # noqa
_temp = sys.path.pop(0)  # noqa
# sys.path = sys.path[1:]  # noqa

import numpy as np
import pytest
from scipy.interpolate import CubicSpline, interp1d

print(sys.path)
# sys.path.insert(0, _temp)
print(sys.path)
from cached_interpolate import CachingInterpolant

print(sys.path)
sys.path.insert(0, _temp)
print(sys.path)


def test_cubic_matches_scipy():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = CachingInterpolant(x_values, y_values, kind="cubic")
    test_points = np.random.uniform(0, 1, 10000)
    max_diff = 0
    for _ in range(100):
        y_values = np.random.uniform(-1, 1, 10)
        scs = CubicSpline(x=x_values, y=y_values, bc_type="natural")
        diffs = spl(test_points, y=y_values) - scs(test_points)
        max_diff = max(np.max(diffs), max_diff)
    assert max_diff, 1e-10


def test_nearest_matches_scipy():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = CachingInterpolant(x_values, y_values, kind="nearest")
    test_points = np.random.uniform(0, 1, 10000)
    max_diff = 0
    for _ in range(100):
        y_values = np.random.uniform(-1, 1, 10)
        scs = interp1d(x=x_values, y=y_values, kind="nearest")
        diffs = spl(test_points, y=y_values) - scs(test_points)
        max_diff = max(np.max(diffs), max_diff)
    assert max_diff < 1e-10


def test_linear_matches_numpy():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = CachingInterpolant(x_values, y_values, kind="linear")
    test_points = np.random.uniform(0, 1, 10000)
    max_diff = 0
    for _ in range(100):
        y_values = np.random.uniform(-1, 1, 10)
        npy = np.interp(test_points, x_values, y_values)
        diffs = spl(test_points, y=y_values) - npy
        max_diff = max(np.max(diffs), max_diff)
    assert max_diff < 1e-10


def test_single_input():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = CachingInterpolant(x_values, y_values, kind="linear")
    assert spl(0) == y_values[0]


def test_single_complex():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    y_values = y_values + 1j * (1 - y_values)
    spl = CachingInterpolant(x_values, y_values, kind="linear")
    assert spl(0) == y_values[0]


def test_interpolation_at_lower_bound():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = CachingInterpolant(x_values, y_values, kind="linear")
    test_point = 0
    assert abs(spl(test_point) - y_values[0]) < 1e-5


def test_interpolation_at_upper_bound():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = CachingInterpolant(x_values, y_values, kind="linear")
    test_point = 1
    assert abs(spl(test_point) - y_values[-1]) < 1e-5


def test_bad_interpolation_method_raises_error():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    with pytest.raises(ValueError):
        _ = CachingInterpolant(x_values, y_values, kind="bad method")


def test_running_without_new_y_values():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    spl = CachingInterpolant(x_values, y_values, kind="cubic")
    old_values = spl._data
    _ = spl(np.array([0, 1]), y=np.random.uniform(-1, 1, 10), use_cache=False)
    assert np.max(old_values - spl._data) > 1e-5


def test_running_with_complex_input_linear():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    y_values = y_values * np.exp(1j * np.random.uniform(0, 2 * np.pi, 10))
    spl = CachingInterpolant(x_values, y_values, kind="linear")
    scs = interp1d(x=x_values, y=y_values, kind="linear")
    test_points = np.random.uniform(0, 1, 10)
    scs_test = scs(test_points)
    diffs = spl(test_points) - scs_test
    assert np.max(diffs) < 1e-10


def test_running_with_complex_input_cubic():
    x_values = np.linspace(0, 1, 10)
    y_values = np.random.uniform(-1, 1, 10)
    y_values = y_values * np.exp(1j * np.random.uniform(0, 2 * np.pi, 10))
    spl = CachingInterpolant(x_values, y_values, kind="cubic")
    scs = CubicSpline(x=x_values, y=y_values, bc_type="natural")
    test_points = np.random.uniform(0, 1, 10)
    scs_test = scs(test_points)
    diffs = spl(test_points) - scs_test
    assert np.max(diffs) < 1e-10


@pytest.mark.parametrize("kind", ["nearest", "linear", "cubic"])
def test_2d_input(kind):
    kwargs = dict(x=np.linspace(0, 1, 5), y=np.random.uniform(0, 1, 5), kind=kind)
    test_values = np.random.uniform(0, 1, (2, 10000))
    spl = CachingInterpolant(**kwargs)
    array_test = spl(test_values)
    loop_test = list()
    for ii in range(2):
        spl = CachingInterpolant(**kwargs)
        loop_test.append(spl(test_values[ii]))
    loop_test = np.array(loop_test)
    assert np.array_equal(loop_test, array_test)
