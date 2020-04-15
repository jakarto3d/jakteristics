# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np

cimport cython
from libc.math cimport fabs
from scipy.linalg.cython_blas cimport dsyrk
from scipy.linalg.cython_lapack cimport dsyev

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter](Iter first, Iter last)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void c_matmul_transposed(double[::1, :] a, double[::1, :] out, double alpha) nogil:
    cdef:
        char *uplo = 'U'
        char *trans = 'N'
        int n = a.shape[0]
        int k = a.shape[1]
        int lda = a.strides[1] // sizeof(double)  # for column major
        int ldc = out.strides[1] // sizeof(double)  # for column major
        double beta = 0.0

    dsyrk(uplo, trans, &n, &k, &alpha, &a[0,0], &lda, &beta, &out[0,0], &ldc)

    cdef int i, j

    # dsyrk computes results for the upper matrix only
    for i in range(n):
        for j in range(n):
            if i > j:
                out[i, j] = out[j, i]


def py_matmul_transposed(a):
    out = np.empty((a.shape[0], a.shape[0]), dtype=np.float64, order="F")
    c_matmul_transposed(np.asfortranarray(a), out, 1.0)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline void substract_mean(double[::1, :] a) nogil:
    """a must be of shape (3, n)"""
    cdef:
        int n_cols = a.shape[1]
        double *mean = [0, 0, 0]
        int c
    
    for c in range(n_cols):
        mean[0] += a[0, c]
        mean[1] += a[1, c]
        mean[2] += a[2, c]

    mean[0] /= n_cols
    mean[1] /= n_cols
    mean[2] /= n_cols

    for c in range(n_cols):
        a[0, c] -= mean[0]
        a[1, c] -= mean[1]
        a[2, c] -= mean[2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void c_covariance(double[::1, :] a, double[::1, :] out) nogil:
    """a must be of shape (3, n)"""
    cdef:
        int n_rows = a.shape[0]
        int i, j
        double fact

    substract_mean(a)

    fact = 1.0 / (a.shape[1] - 1)

    c_matmul_transposed(a, out, fact)
    

def py_covariance(a):
    out = np.empty((a.shape[0], a.shape[0]), dtype=np.float64, order="F")
    c_covariance(np.asfortranarray(a), out)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void c_eigenvectors(double[::1, :] a, double[:] eigenvalues) nogil:
    """
    http://www.netlib.org/lapack/explore-html/d3/d88/group__real_s_yeigen_ga63d8d12aef8f2711d711d9e6bd833e46.html#ga63d8d12aef8f2711d711d9e6bd833e46
    """
    cdef:
        char *jobz = 'V'  # compute eigenvalues and eigenvectors
        char *uplo = 'U'  # upper triangle of A is stored
        int n = a.shape[0]
        int lda = a.strides[1] // sizeof(double)  # for column major
        double work[256]  # not sure how to properly set the size of this array
        int lwork = 256
        int info

    dsyev(jobz, uplo, &n, &a[0,0], &lda, &eigenvalues[0], &work[0], &lwork, &info)

    c_sort_eigenvalues(a, eigenvalues)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_sort_eigenvalues(double[::1, :] eigenvectors, double[:] eigenvalues) nogil:
    """Sort eigenvalues and eigenvectors"""
    cdef:
        int i
        int j
        int k
        int n = 3
        double eigenvalues_copy[3]
        double eigenvectors_copy[3][3]

    for i in range(n):
        eigenvalues[i] = fabs(eigenvalues[i])

        eigenvalues_copy[i] = eigenvalues[i]
        for j in range(n):
            eigenvectors_copy[i][j] = eigenvectors[i, j]

    sort(&eigenvalues[0], (&eigenvalues[0]) + n)
    # reverse
    cdef double temp = eigenvalues[2]
    eigenvalues[2] = eigenvalues[0]
    eigenvalues[0] = temp

    for i in range(n):
        for j in range(n):
            if fabs(eigenvalues[i] - eigenvalues_copy[j]) < 1e-5:
                for k in range(n):
                    # eigenvectors is column-major, and eigenvectors_copy is row major
                    eigenvectors[i, k] = eigenvectors_copy[k][j]
                break


def py_eigenvectors(a):
    eigenvalues = np.empty(a.shape[0], dtype=np.float64, order="F")
    a = np.asfortranarray(a)
    c_eigenvectors(a, eigenvalues)
    return eigenvalues, a