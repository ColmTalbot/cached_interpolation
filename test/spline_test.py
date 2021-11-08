import unittest

import numpy as np
from scipy.interpolate import interp1d, CubicSpline

from cached_spline import CachingInterpolant


class SplineTest(unittest.TestCase):

    def setUp(self) -> None:
        self.x_values = np.linspace(0, 1, 10)
        self.y_values = np.random.uniform(-1, 1, 10)

    def tearDown(self) -> None:
        pass

    def test_cubic_matches_scipy(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="cubic")
        scs = CubicSpline(x=self.x_values, y=self.y_values, bc_type="natural")
        test_points = np.random.uniform(0, 1, 10000)
        diffs = spl(test_points) - scs(test_points)
        self.assertLess(np.max(diffs), 1e-10)

    def test_nearest_matches_scipy(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="nearest")
        scs = interp1d(x=self.x_values, y=self.y_values, kind="nearest")
        test_points = np.random.uniform(0, 1, 10000)
        diffs = spl(test_points) - scs(test_points)
        self.assertLess(np.max(diffs), 1e-10)

    def test_linear_matches_numpy(self):
        spl = CachingInterpolant(self.x_values, self.y_values, kind="linear")
        test_points = np.random.uniform(0, 1, 10000)
        diffs = spl(test_points) - np.interp(test_points, self.x_values, self.y_values)
        self.assertLess(np.max(diffs), 1e-10)
