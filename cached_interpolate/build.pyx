cimport numpy as np  # noqa
import numpy as np


cpdef build_linear_interpolant(xx, yy):
    aa = yy[:len(xx) - 1]
    bb = np.diff(yy) / np.diff(xx)
    return aa, bb


cpdef build_natural_cubic_spline(xx, yy):
    cdef int ii, jj
    cdef int n_points = len(xx) - 1
    cdef double ll, _c

    aa = yy.copy()[:n_points]
    bb = np.empty(n_points)
    cc = np.empty(n_points)
    dd = np.empty(n_points)

    delta = np.diff(xx)

    alpha = 3 * np.diff(np.diff(yy) / delta)

    mu = np.empty(n_points + 1)
    zz = np.empty(n_points)

    cdef double[:] a_ = aa
    cdef double[:] b_ = bb
    cdef double[:] c_ = cc
    cdef double[:] d_ = dd
    cdef double[:] m_ = mu
    cdef double[:] x_ = xx
    cdef double[:] z_ = zz
    cdef double[:] al = alpha
    cdef double[:] de = delta

    m_[0] = 0
    z_[0] = 0

    for ii in range(1, n_points):
        ll = 2 * (x_[ii + 1] - x_[ii - 1]) - de[ii - 1] * m_[ii - 1]
        m_[ii] = de[ii] / ll
        z_[ii] = (al[ii - 1] - de[ii - 1] * z_[ii - 1]) / ll

    _c = 0

    for jj in range(n_points - 1, -1, -1):
        c_[jj] = z_[jj] - m_[jj] * _c
        b_[jj] = (a_[jj + 1] - a_[jj]) / de[jj] - de[jj] * (_c + 2 * c_[jj]) / 3
        d_[jj] = (_c - c_[jj]) / 3 / de[jj]
        _c = c_[jj]

    return aa, bb, cc, dd
