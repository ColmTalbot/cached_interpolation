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
