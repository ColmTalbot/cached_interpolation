import jax.numpy as jnp
from jax import jit


@jit
def build_linear_interpolant(xx, yy):
    aa = yy[: len(xx) - 1]
    bb = jnp.diff(yy) / jnp.diff(xx)
    return aa, bb


@jit
def build_natural_cubic_spline(xx, yy):
    n_points = len(xx) - 1
    aa = yy.copy()[:n_points]

    delta = jnp.diff(xx)
    alpha = 3 * jnp.diff(jnp.diff(yy) / delta)

    mu = jnp.zeros_like(yy)
    zz = jnp.zeros_like(aa)
    for ii in range(1, n_points):
        ll = 2 * (xx[ii + 1] - xx[ii - 1]) - delta[ii - 1] * mu[ii - 1]
        mu = mu.at[ii].set(delta[ii] / ll)
        zz = zz.at[ii].set((alpha[ii - 1] - delta[ii - 1] * zz[ii - 1]) / ll)

    bb = jnp.zeros_like(aa)
    cc = jnp.zeros_like(aa)
    dd = jnp.zeros_like(aa)
    c_old = 0
    for jj in range(n_points - 1, -1, -1):
        cc = cc.at[jj].set(zz[jj] - mu[jj] * c_old)
        bb = bb.at[jj].set(
            (yy[jj + 1] - yy[jj]) / delta[jj] - delta[jj] * (c_old + 2 * cc[jj]) / 3
        )
        dd = dd.at[jj].set((c_old - cc[jj]) / 3 / delta[jj])
        c_old = cc[jj]

    return aa, bb, cc, dd
