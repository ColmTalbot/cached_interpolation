import sys

print(sys.path)  # noqa
sys.path = sys.path[1:]  # noqa

import unittest

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from cached_interpolate import CachingInterpolant


class SplineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x_values = np.linspace(0, 1, 10)
        self.y_values = np.random.uniform(-1, 1, 10)

    def tearDown(self) -> None:
        pass

    def test_cubic_matches_scipy(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="cubic")
        test_points = np.random.uniform(0, 1, 10000)
        max_diff = 0
        for _ in range(100):
            y_values = np.random.uniform(-1, 1, 10)
            scs = CubicSpline(x=self.x_values, y=y_values, bc_type="natural")
            diffs = spl(test_points, y=y_values) - scs(test_points)
            max_diff = max(np.max(diffs), max_diff)
        self.assertLess(max_diff, 1e-10)

    def test_nearest_matches_scipy(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="nearest")
        test_points = np.random.uniform(0, 1, 10000)
        max_diff = 0
        for _ in range(100):
            y_values = np.random.uniform(-1, 1, 10)
            scs = interp1d(x=self.x_values, y=y_values, kind="nearest")
            diffs = spl(test_points, y=y_values) - scs(test_points)
            max_diff = max(np.max(diffs), max_diff)
        self.assertLess(max_diff, 1e-10)

    def test_linear_matches_numpy(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="linear")
        test_points = np.random.uniform(0, 1, 10000)
        max_diff = 0
        for _ in range(100):
            y_values = np.random.uniform(-1, 1, 10)
            npy = np.interp(test_points, self.x_values, y_values)
            diffs = spl(test_points, y=y_values) - npy
            max_diff = max(np.max(diffs), max_diff)
        self.assertLess(max_diff, 1e-10)

    def test_single_input(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="linear")
        self.assertEqual(spl(0), self.y_values[0])

    def test_single_complex(self):
        y_values = self.y_values + 1j * (1 - self.y_values)
        spl = CachingInterpolant(self.x_values, y_values, kind="linear")
        self.assertEqual(spl(0), y_values[0])

    def test_interpolation_at_lower_bound(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="linear")
        test_point = 0
        self.assertAlmostEqual(spl(test_point), self.y_values[0], 5)

    def test_interpolation_at_upper_bound(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="linear")
        test_point = 1
        self.assertAlmostEqual(spl(test_point), self.y_values[-1], 5)

    def test_bad_interpolation_method_raises_error(self):
        with self.assertRaises(ValueError):
            _ = CachingInterpolant(self.x_values, self.y_values, kind="bad method")

    def test_running_without_new_y_values(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="cubic")
        old_values = spl._data
        _ = spl(np.array([0, 1]), y=np.random.uniform(-1, 1, 10), use_cache=False)
        self.assertGreater(np.max(old_values - spl._data), 1e-5)

    def test_running_with_complex_input_linear(self):
        y_values = self.y_values * np.exp(1j * np.random.uniform(0, 2 * np.pi, 10))
        spl = CachingInterpolant(self.x_values, y_values, kind="linear")
        scs = interp1d(x=self.x_values, y=y_values, kind="linear")
        test_points = np.random.uniform(0, 1, 10)
        scs_test = scs(test_points)
        diffs = spl(test_points) - scs_test
        self.assertLess(np.max(diffs), 1e-10)

    def test_running_with_complex_input_cubic(self):
        y_values = self.y_values * np.exp(1j * np.random.uniform(0, 2 * np.pi, 10))
        spl = CachingInterpolant(self.x_values, y_values, kind="cubic")
        scs = CubicSpline(x=self.x_values, y=y_values, bc_type="natural")
        test_points = np.random.uniform(0, 1, 10)
        scs_test = scs(test_points)
        diffs = spl(test_points) - scs_test
        self.assertLess(np.max(diffs), 1e-10)
