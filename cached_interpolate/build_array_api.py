import array_api_compat as xpc
import array_api_extra as xpx


def build_linear_interpolant(xx, yy):
    xp = xpc.array_namespace(xx, yy)
    aa = yy[: len(xx) - 1]
    bb = xp.diff(yy) / xp.diff(xx)
    return aa, bb


def build_natural_cubic_spline(xx, yy):
    xp = xpc.array_namespace(xx, yy)
    n_points = len(xx) - 1
    aa = xp.asarray(yy[:n_points])

    delta = xp.diff(xx)
    alpha = 3 * xp.diff(xp.diff(yy) / delta)

    mu = xp.zeros_like(yy)
    zz = xp.zeros_like(aa)
    for ii in range(1, n_points):
        ll = 2 * (xx[ii + 1] - xx[ii - 1]) - delta[ii - 1] * mu[ii - 1]
        mu = xpx.at(mu, ii).set(delta[ii] / ll, xp=xp)
        zz = xpx.at(zz, ii).set((alpha[ii - 1] - delta[ii - 1] * zz[ii - 1]) / ll, xp=xp)

    bb = xp.zeros_like(aa)
    cc = xp.zeros_like(aa)
    dd = xp.zeros_like(aa)
    c_old = 0
    for jj in range(n_points - 1, -1, -1):
        cc = xpx.at(cc, jj).set(zz[jj] - mu[jj] * c_old, xp=xp)
        bb = xpx.at(bb, jj).set(
            (yy[jj + 1] - yy[jj]) / delta[jj] - delta[jj] * (c_old + 2 * cc[jj]) / 3,
            xp=xp,
        )
        dd = xpx.at(dd, jj).set((c_old - cc[jj]) / 3 / delta[jj], xp=xp)
        c_old = cc[jj]

    return aa, bb, cc, dd


try:
    import jax

    build_linear_interpolant = jax.jit(build_linear_interpolant)
    build_natural_cubic_spline = jax.jit(build_natural_cubic_spline)
except ImportError:
    pass
