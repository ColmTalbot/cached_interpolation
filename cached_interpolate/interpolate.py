from numbers import Number

import numpy as np

from .build import build_linear_interpolant, build_natural_cubic_spline


def to_numpy(array):
    """
    Convert an array to a numpy array.
    Numeric types and pandas objects are returned unchanged.

    Parameters
    ==========
    array: array-like
        The array to convert.
    """
    if isinstance(array, (Number, np.ndarray)):
        return array
    elif "cupy" in array.__class__.__module__:
        from cupy import asnumpy

        return asnumpy(array)
    elif "pandas" in array.__class__.__module__:
        return array
    elif "jax" in array.__class__.__module__:
        return np.asarray(array)
    else:
        raise TypeError(f"Cannot convert {type(array)} to numpy array")


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

    from cached_interpolate import CachingInterpolant

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

    If you have access to an `nvidia` GPU and are evaluating the spline at ~ O(10^5) or more points you may want
    to switch to the `cupy` backend.
    This uses `cupy` just for the evaluation stage, not for computing the interpolation coefficients.

    ```python
    import cupy as cp

    evaluation_points = cp.asarray(evaluation_points)

    interpolant = CachingInterpolant(x=x_nodes, y=y_nodes, backend=cp)
    interpolated_values = interpolant(evaluation_points)
    ```
    """

    def __init__(self, x, y, kind="cubic", backend=np, bc_type="natural"):
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
        if bc_type != "natural":
            raise NotImplementedError(
                "Only natural boundary conditions are supported for the generic interpolant"
            )
        self.return_float = False
        self.bk = backend
        allowed_kinds = ["nearest", "linear", "cubic"]
        if kind not in allowed_kinds:
            raise ValueError(f"kind must be in {allowed_kinds}")
        self.x_array = x
        self.y_array = y
        self._data = None
        self.kind = kind
        self._cached = False

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = kind
        data = self.build()
        if data is not None:
            data = self.bk.asarray(list(data))
        self._data = data

    def build(self):
        """
        Call the constructor for the interpolant.

        :return: tuple
            Tuple containing the interpolation coefficients
        """
        if self.kind == "cubic":
            if self.bk.__name__ in ["numpy", "cupy"]:
                builder = build_natural_cubic_spline
            elif self.bk.__name__ == "jax.numpy":
                from .build_jax import build_natural_cubic_spline as builder
        elif self.kind == "linear":
            if self.bk.__name__ in ["numpy", "cupy"]:
                builder = build_linear_interpolant
            elif self.bk.__name__ == "jax.numpy":
                from .build_jax import build_linear_interpolant as builder
        elif self.kind == "nearest":
            return self.bk.asarray(self.y_array)
        return self.bk.asarray(builder(xx=self.x_array, yy=self.y_array))

    def _construct_cache(self, x_values):
        """
        Calculate the quantities required for the interpolation.

        These are:
        - the indices of the reference x node.
        - the distance from that node along with the required powers of that distance.

        This internally uses numpy arrays and then converts the generated quantities
        to the appropriate backend at the end.

        :param x_values: ndarray
            The values that the interpolant will be evaluated at
        """
        x_array = to_numpy(self.x_array)
        x_values = np.atleast_1d(to_numpy(x_values))
        if x_values.size == 1:
            self.return_float = True
        input_shape = x_values.shape
        x_values = x_values.reshape(-1)
        self._cached = True
        if self.kind == "nearest":
            self._idxs = np.asarray(
                [np.argmin(abs(xval - x_array)) for xval in x_values]
            ).reshape(input_shape)
        else:
            self._idxs = np.asarray(
                [
                    0 if xval <= x_array[0] else np.where(xval > x_array)[0][-1].item()
                    for xval in x_values
                ]
            ).reshape(input_shape)
            x_values = x_values.reshape(input_shape)
            diffs = [np.ones(x_values.shape), x_values - x_array[self._idxs]]
            if self.kind == "cubic":
                diffs += [
                    (x_values - x_array[self._idxs]) ** 2,
                    (x_values - x_array[self._idxs]) ** 3,
                ]
                self._diffs = np.stack(diffs)
            self._diffs = self.bk.asarray(diffs)
        self._idxs = self.bk.asarray(self._idxs)

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
            self._data = self.build()
        if not (self._cached and use_cache):
            self._construct_cache(x_values=x)
        if self.kind == "cubic":
            out = self._call_cubic()
        elif self.kind == "linear":
            out = self._call_linear()
        elif self.kind == "nearest":
            out = self._call_nearest()
        if self.return_float:
            out = out[0]
        return out

    def _call_nearest(self):
        return self._data[self._idxs]

    def _call_linear(self):
        return self.bk.sum(self._data[:, self._idxs] * self._diffs, axis=0)

    def _call_cubic(self):
        return self.bk.sum(self._data[:, self._idxs] * self._diffs, axis=0)


class RegularCachingInterpolant:
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

    from cached_interpolate import CachingInterpolant

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

    If you have access to an `nvidia` GPU and are evaluating the spline at ~ O(10^5) or more points you may want
    to switch to the `cupy` backend.
    This uses `cupy` just for the evaluation stage, not for computing the interpolation coefficients.

    ```python
    import cupy as cp

    evaluation_points = cp.asarray(evaluation_points)

    interpolant = CachingInterpolant(x=x_nodes, y=y_nodes, backend=cp)
    interpolated_values = interpolant(evaluation_points)
    ```
    """

    def __init__(self, x, y, kind="cubic", backend=np, bc_type="not-a-knot"):
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
        from .matrix_forms import MAPPING

        self.bk = backend
        self.n_nodes = len(x)
        if bc_type not in MAPPING:
            raise NotImplementedError(
                f"bc_type must be one of {list(MAPPING.keys())} not {bc_type}"
            )
        self.conversion = self.bk.asarray(MAPPING[bc_type](self.n_nodes))
        self.return_float = False
        allowed_kinds = ["nearest", "linear", "cubic"]
        if kind not in allowed_kinds:
            raise ValueError(f"kind must be in {allowed_kinds}")
        self.x_array = self.bk.asarray(x)
        self.y_array = self.bk.asarray(y)
        self._data = None
        self.kind = kind
        self._cached = False
        self.delta = x[1] - x[0]

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = kind
        data = self.build()
        if data is not None:
            data = self.bk.asarray(list(data))
        self._data = data

    def build(self):
        """
        Call the constructor for the interpolant.

        :return: tuple
            Tuple containing the interpolation coefficients
        """

        if self.kind == "cubic":
            values = self.conversion @ self.y_array
            return self.bk.asarray(
                [
                    self.y_array[:-1],
                    self.y_array[1:],
                    values[:-1],
                    values[1:],
                ]
            )
        elif self.kind == "linear":
            return self.bk.asarray(
                [
                    self.y_array[: self.n_nodes - 1],
                    self.bk.diff(self.y_array) / self.bk.diff(self.x_array),
                ]
            )
        elif self.kind == "nearest":
            return self.bk.asarray(self.y_array)

    def _construct_cache(self, x_values):
        """
        Calculate the quantities required for the interpolation.

        These are:
        - the indices of the reference x node.
        - the distance from that node along with the required powers of that distance.

        This internally uses numpy arrays and then converts the generated quantities
        to the appropriate backend at the end.

        :param x_values: np.ndarray
            The values that the interpolant will be evaluated at
        """
        x_array = to_numpy(self.x_array)
        x_values = np.atleast_1d(to_numpy(x_values))

        if x_values.size == 1:
            self.return_float = True

        scaled = (x_values - x_array[0]) / self.delta
        if self.kind == "nearest":
            idxs = np.clip(
                np.round(scaled).astype(int), a_min=0, a_max=self.n_nodes - 1
            )
        else:
            idxs = np.clip(
                np.floor(scaled).astype(int), a_min=0, a_max=self.n_nodes - 2
            )
        self._idxs = self.bk.asarray(idxs)
        if self.kind == "cubic":
            bb = scaled - idxs
            aa = 1 - bb
            cc = (aa**3 - aa) / 6
            dd = (bb**3 - bb) / 6
            self._diffs = self.bk.asarray([aa, bb, cc, dd])
        elif self.kind == "linear":
            self._diffs = self.bk.asarray(
                [np.ones(x_values.shape), x_values - x_array[idxs]]
            )
        self._cached = True

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
            self._data = self.build()
        if not (self._cached and use_cache):
            self._construct_cache(x_values=x)
        if self.kind == "cubic":
            out = self._call_cubic()
        elif self.kind == "linear":
            out = self._call_linear()
        elif self.kind == "nearest":
            out = self._call_nearest()
        if self.return_float:
            out = out[0]
        return out

    def _call_nearest(self):
        return self._data[self._idxs]

    def _call_linear(self):
        return self.bk.sum(self._data[:, self._idxs] * self._diffs, axis=0)

    def _call_cubic(self):
        return self.bk.sum(self._data[:, self._idxs] * self._diffs, axis=0)
