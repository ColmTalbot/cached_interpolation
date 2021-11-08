import numpy as np

from .build import build_linear_spline, build_natural_cubic_spline


class CachingInterpolant:
    """
    Efficient evaluation of interpolants at fixed points.

    Evaluating interpolants typically requires two stages:
    1. finding the closest knot of the interpolant to the new point and the distance from that knot.
    2. evaluating the interpolant at that point.

    Sometimes it is necessary to evaluate many interpolants with identical knot points and evaluation
    points but different functions being approximated and so the first of these stages is done many times unnecessarily.
    This can be made more efficient by caching the locations of the evaluation points leaving just the evaluation of the
    interpolation coefficients to be done at each iteration.

    A further advantage of this, is that it allows broadcasting the interpolation using `cupy`.

    This package implements this caching for nearest neighbour, linear, and cubic interpolation.

    ```python
    import numpy as np

    from cached_spline import CachingInterpolant

    x_nodes = np.linspace(0, 1, 10)
    y_nodes = np.random.uniform(-1, 1, 10)
    evaluation_points = np.random.uniform(0, 1, 10000)

    interpolant = CachingInterpolant(x=x_nodes, y=y_nodes, kind="cubic")
    interpolated_values = interpolant(evaluation_points)
    ```

    We can now evaluate this interpolant in a loop with the caching.

    ```python
    for _ in range(1000):
        y_nodes = np.random.uniform(-1, 1, 10)
        interpolant(x=evaluation_points, y=y_nodes)
    ```

    If we need to evaluate for a new set of points, we have to tell the interpolant to reset the cache.
    There are two ways to do this:
    - create a new interpolant, this will require reevaluating the interplation coefficients.
    - disable the evaluation point caching.

    ```python
    new_evaluation_points = np.random.uniform(0, 1, 10000)
    interpolant(x=new_evaluation_points, use_cache=False)
    ```

    """

    def __init__(self, x, y, kind="cubic", backend=np):
        """
        Initialize the interpolator

        :param x: np.ndarray
            The nodes of the interpolant
        :param y: np.ndarray
            The value of the function being interpolated at the nodes
        :param kind: str
            The interpolation type, should be in ["nearest", "linear", "cubic"],
            default="cubic"
        :param backend: module
            Backend for array operations, e.g., `numpy` or `cupy`.
            This enables simple GPU acceleration.
        """
        allowed_kinds = ["nearest", "linear", "cubic"]
        if kind not in allowed_kinds:
            raise ValueError(f"kind must be in {allowed_kinds}")
        self.x_array = x
        self.y_array = y
        self._spline_data = None
        self.kind = kind
        self.bk = backend
        self._cached = False

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = kind
        self._spline_data = self.build()

    def build(self):
        """
        Call the constructor for the interpolant.

        :return: tuple
            Tuple containing the interpolation coefficients
        """
        if self.kind == "cubic":
            return build_natural_cubic_spline(xx=self.x_array, yy=self.y_array)
        elif self.kind == "linear":
            return build_linear_spline(xx=self.x_array, yy=self.y_array)

    def _construct_cache(self, x_values):
        """
        Calculate the quantities required for the interpolation.

        These are:
        - the indices of the reference x node.
        - the distance from that node along with the required powers of that distance.

        :param x_values: np.ndarray
            The values that the interpolant will be evaluated at
        """
        self._cached = True
        self._idxs = self.bk.empty(x_values.shape, dtype=int)
        if self.kind == "nearest":
            for ii, xval in enumerate(x_values):
                self._idxs[ii] = np.argmin(abs(xval - self.x_array))
        else:
            for ii, xval in enumerate(x_values):
                self._idxs[ii] = self.bk.where(xval > self.x_array)[0][-1]
            self._diffs = x_values - self.x_array[self._idxs]
            if self.kind == "cubic":
                self._diffs2 = self._diffs ** 2
                self._diffs3 = self._diffs ** 3

    def __call__(self, x, y=None, use_cache=True):
        """
        Call the interpolant with desired caching

        :param x: np.ndarray
            The values that the interpolant will be evaluated at
        :param y: np.ndarray
            New interpolation points, this disables the caching of the target function
        :param use_cache: bool
            Whether to use the cached x values
        :return: np.ndarray
            The value of the interpolant at `x`
        """
        if y is not None:
            self.y_array = y
            self.build()
        if not (self._cached and use_cache):
            self._construct_cache(x_values=x)
        if self.kind == "cubic":
            return self._call_cubic()
        elif self.kind == "linear":
            return self._call_linear()
        elif self.kind == "nearest":
            return self._call_nearest()

    def _call_nearest(self):
        return self.y_array[self._idxs]

    def _call_linear(self):
        output = self._spline_data[0][self._idxs]
        output += self._spline_data[1][self._idxs] * self._diffs
        return output

    def _call_cubic(self):
        output = self._call_linear()
        output += self._spline_data[2][self._idxs] * self._diffs2
        output += self._spline_data[3][self._idxs] * self._diffs3
        return output