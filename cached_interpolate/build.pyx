cimport numpy as np  # noqa
import numpy as np

ctypedef fused numeric:
    complex
    double


cpdef build_linear_interpolant(xx, yy):
    aa = yy[:len(xx) - 1]
    bb = np.diff(yy) / np.diff(xx)
    return aa, bb


cpdef build_natural_cubic_spline(double[:] xx, numeric[:] yy):
    cdef int ii, jj
    cdef int n_points = len(xx) - 1
    cdef numeric ll, _c

    aa = yy.copy()[:n_points]
    bb = np.empty_like(aa)
    cc = np.empty_like(aa)
    dd = np.empty_like(aa)

    delta = np.diff(xx)

    alpha = 3 * np.diff(np.diff(yy) / delta)

    mu = np.empty_like(yy)
    zz = np.empty_like(aa)

    cdef numeric[:] a_ = aa
    cdef numeric[:] b_ = bb
    cdef numeric[:] c_ = cc
    cdef numeric[:] d_ = dd
    cdef numeric[:] m_ = mu
    cdef double[:] x_ = xx
    cdef numeric[:] z_ = zz
    cdef numeric[:] al = alpha
    cdef double[:] de = delta

    m_[0] = 0.0
    z_[0] = 0.0

    for ii in range(1, n_points):
        ll = 2 * (x_[ii + 1] - x_[ii - 1]) - de[ii - 1] * m_[ii - 1]
        m_[ii] = de[ii] / ll
        z_[ii] = (al[ii - 1] - de[ii - 1] * z_[ii - 1]) / ll

    _c = 0.0

    for jj in range(n_points - 1, -1, -1):
        c_[jj] = z_[jj] - m_[jj] * _c
        b_[jj] = (
            (yy[jj + 1] - yy[jj]) / de[jj]
            - de[jj] * (_c + 2 * c_[jj]) / 3
        )
        d_[jj] = (_c - c_[jj]) / 3 / de[jj]
        _c = c_[jj]

    return np.array(aa), bb, cc, dd
