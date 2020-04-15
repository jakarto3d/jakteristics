# cython: language_level=3
# distutils: language = c++

cdef void c_matmul_transposed(double[::1, :] a, double[::1, :] out, double alpha) nogil
cdef void c_covariance(double[::1, :] a, double[::1, :] out) nogil
cdef void c_eigenvectors(double[::1, :] a, double[:] eigenvalues) nogil
