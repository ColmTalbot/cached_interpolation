import numpy as np

from ._spline import build_linear_spline, build_natural_cubic_spline


class CachingSpline:

    def __init__(self, x_array, y_array, kind="cubic", backend=np):
        self.x_array = x_array
        self.y_array = y_array
        self.kind = kind
        self.n_points = len(self.x_array) - 1
        self._spline_data = self.build()
        self.bk = backend
        self._cached = False

    def build(self):
        if self.kind == "cubic":
            return build_natural_cubic_spline(xx=self.x_array, yy=self.y_array)
        elif self.kind == "linear":
            return build_linear_spline(xx=self.x_array, yy=self.y_array)

    def _construct_cache(self, x_values):
        self._cached = True
        self._idxs = self.bk.empty(x_values.shape, dtype=int)
        self._diffs = self.bk.empty(x_values.shape)
        for ii, xval in enumerate(x_values):
            self._idxs[ii] = self.bk.where(xval > self.x_array)[0][-1]
            self._diffs[ii] = xval - self.x_array[self._idxs[ii]]
        if self.kind == "cubic":
            self._diffs2 = self._diffs ** 2
            self._diffs3 = self._diffs ** 3

    def __call__(self, x_values, y_values=None, use_cache=True):
        if y_values is not None:
            self.y_array = y_values
            self.build()
        if not (self._cached and use_cache):
            self._construct_cache(x_values=x_values)
        if self.kind == "cubic":
            return self._call_cubic()
        elif self.kind == "linear":
            return self._call_linear()

    def _call_linear(self):
        output = self._spline_data[0][self._idxs]
        output += self._spline_data[1][self._idxs] * self._diffs
        return output

    def _call_cubic(self):
        output = self._call_linear()
        output += self._spline_data[2][self._idxs] * self._diffs2
        output += self._spline_data[3][self._idxs] * self._diffs3
        return output
