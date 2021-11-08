# cached_interpolate
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