# cython: language_level=3
# distutils: language = c++

import numpy as np
import multiprocessing

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memset
from libcpp cimport bool
cimport openmp
cimport cython
from cython.parallel import prange, parallel
from libc.math cimport fabs, pow, log, sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap
from libcpp.string cimport string
from libc.stdint cimport uintptr_t, uint32_t, int8_t, uint8_t, int64_t

from .ckdtree.ckdtree cimport cKDTree, ckdtree, query_ball_point
from . cimport utils
from .constants import FEATURE_NAMES


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def compute_features(
    double [:, ::1] points,
    float search_radius,
    *,
    cKDTree kdtree=None,
    int num_threads=-1,
    int max_k_neighbors=50000,
    bint euclidean_distance=True,
    feature_names=FEATURE_NAMES,
    float eps=0.0,
):
    cdef:
        cppmap [string, uint8_t] features_map

        int64_t n_points = points.shape[0]
        double [::1, :] neighbor_points
        double [::1, :] eigenvectors
        double [:] eigenvalues

        int i, j, k
        uint32_t neighbor_id
        uint32_t n_neighbors_at_id
        int thread_id

        float [:, :] features = np.zeros((n_points, len(feature_names)), dtype=np.float32)

        const np.float64_t[:, ::1] radius_vector
        np.float64_t p = 2 if euclidean_distance else 1
        np.float64_t eps_scipy = 0.0
        vector[np.intp_t] *** threaded_vvres
        int return_length = <int> False

    if not points.shape[1] == 3:
        raise ValueError("You must provide an (n x 3) numpy array.")

    if num_threads == -1:
        num_threads = multiprocessing.cpu_count()

    if kdtree is None:
        kdtree = cKDTree(points)

    for n, name in enumerate(feature_names):
        if name not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature name: {name}")
        features_map[name.encode()] = n

    radius_vector = np.full((num_threads, 3), fill_value=search_radius)
    neighbor_points = np.zeros([3, max_k_neighbors * num_threads], dtype=np.float64, order="F")
    eigenvectors = np.zeros([3, 3 * num_threads], dtype=np.float64, order="F")
    eigenvalues = np.zeros(3 * num_threads, dtype=np.float64)

    threaded_vvres = init_result_vectors(num_threads)

    try:
        for i in prange(n_points, nogil=True, num_threads=num_threads):
            thread_id = openmp.omp_get_thread_num()

            threaded_vvres[thread_id][0].clear()
            query_ball_point(
                kdtree.cself,
                &points[i, 0],
                &radius_vector[thread_id, 0],
                p,
                eps_scipy,
                1,
                threaded_vvres[thread_id],
                return_length,
            )

            n_neighbors_at_id = threaded_vvres[thread_id][0].size()

            if n_neighbors_at_id > max_k_neighbors:
                n_neighbors_at_id = max_k_neighbors
            elif n_neighbors_at_id == 0:
                with gil:
                    raise RuntimeError

            for j in range(n_neighbors_at_id):
                neighbor_id = threaded_vvres[thread_id][0][0][j]
                for k in range(3):
                    neighbor_points[k, thread_id * max_k_neighbors + j] = points[neighbor_id, k]

            utils.c_covariance(
                neighbor_points[:, thread_id * max_k_neighbors:thread_id * max_k_neighbors + n_neighbors_at_id],
                eigenvectors[:, thread_id * 3:(thread_id + 1) * 3],
            )
            utils.c_eigenvectors(
                eigenvectors[:, thread_id * 3:(thread_id + 1) * 3],
                eigenvalues[thread_id * 3:(thread_id + 1) * 3],
            )

            compute_features_from_eigenvectors(
                eigenvalues[thread_id * 3 : thread_id * 3 + 3],
                eigenvectors[:, thread_id * 3 : thread_id * 3 + 3],
                features[i, :],
                features_map,
            )

    finally:
        free_result_vectors(threaded_vvres, num_threads)

    return np.asarray(features)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void compute_features_from_eigenvectors(
    double [:] eigenvalues,
    double [:, :] eigenvectors,
    float [:] out,
    cppmap [string, uint8_t] & out_map,
) nogil:
    cdef:
        float l1, l2, l3
        float eigenvalue_sum
        float n0, n1, n2
        float norm

    l1 = eigenvalues[0]
    l2 = eigenvalues[1]
    l3 = eigenvalues[2]

    # Those features are inspired from cloud compare implementation (https://github.com/CloudCompare/CloudCompare/blob/master/CC/src/Neighbourhood.cpp#L871)
    # Those features are also implemented in CGAL (https://doc.cgal.org/4.12/Classification/group__PkgClassificationFeatures.html)

    # Sum of eigenvalues equals the original variance of the data
    eigenvalue_sum = l1 + l2 + l3

    if out_map.count(b"eigenvalue_sum"):
        out[out_map.at(b"eigenvalue_sum")] = eigenvalue_sum

    if out_map.count(b"omnivariance"):
        out[out_map.at(b"omnivariance")] = pow(l1 * l2 * l3, 1.0 / 3.0)

    if out_map.count(b"eigenentropy"):
        out[out_map.at(b"eigenentropy")] = -(l1 * log(l1) + l2 * log(l2) + l3 * log(l3))

    # Anisotropy is the difference between the most principal direction of the point subset.
    # Divided by l1 allows to keep this difference in a ratio between 0 and 1
    # a difference close to zero (l3 close to l1) means that the subset of points are equally spread in the 3 principal directions
    # If the anisotropy is close to 1 (mean l3 close to zero), the subset of points is strongly related only in the first principal component. It depends mainly on one direction.
    if out_map.count(b"anisotropy"):
        out[out_map.at(b"anisotropy")] = (l1 - l3) / l1
    if out_map.count(b"planarity"):
        out[out_map.at(b"planarity")] = (l2 - l3) / l1
    if out_map.count(b"linearity"):
        out[out_map.at(b"linearity")] = (l1 - l2) / l1
    if out_map.count(b"PCA1"):
        out[out_map.at(b"PCA1")] = l1 / eigenvalue_sum
    if out_map.count(b"PCA2"):
        out[out_map.at(b"PCA2")] = l2 / eigenvalue_sum
    # Surface variance is how the third component contributes to the sum of the eigenvalues
    if out_map.count(b"surface_variation"):
        out[out_map.at(b"surface_variation")] = l3 / eigenvalue_sum
    if out_map.count(b"sphericity"):
        out[out_map.at(b"sphericity")] = l3 / l1

    if out_map.count(b"verticality"):
        out[out_map.at(b"verticality")] = 1.0 - fabs(eigenvectors[2, 2])
    
    # eigenvectors is col-major
    if out_map.count(b"nx") or out_map.count(b"ny") or out_map.count(b"nz"):
        n0 = eigenvectors[0, 1] * eigenvectors[1, 2] - eigenvectors[0, 2] * eigenvectors[1, 1]
        n1 = eigenvectors[0, 2] * eigenvectors[1, 0] - eigenvectors[0, 0] * eigenvectors[1, 2]
        n2 = eigenvectors[0, 0] * eigenvectors[1, 1] - eigenvectors[0, 1] * eigenvectors[1, 0]
        norm = sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        if out_map.count(b"nx"):
            out[out_map.at(b"nx")] = n0 / norm
        if out_map.count(b"ny"):
            out[out_map.at(b"ny")] = n1 / norm
        if out_map.count(b"nz"):
            out[out_map.at(b"nz")] = n2 / norm


cdef vector[np.intp_t] *** init_result_vectors(int num_threads):
    """Allocate memory for result vectors, based on thread count"""
    threaded_vvres = <vector[np.intp_t] ***> PyMem_Malloc(num_threads * sizeof(void*))
    if not threaded_vvres:
        raise MemoryError()
    memset(<void*> threaded_vvres, 0, num_threads * sizeof(void*))
    for i in range(num_threads):
        threaded_vvres[i] = <vector[np.intp_t] **> PyMem_Malloc(sizeof(void*))
        if not threaded_vvres[i]:
            raise MemoryError()
        memset(<void*> threaded_vvres[i], 0, sizeof(void*))
        threaded_vvres[i][0] = new vector[np.intp_t]()
    return threaded_vvres


cdef void free_result_vectors(vector[np.intp_t] *** threaded_vvres, int num_threads):
    """Free memory for result vectors"""
    if threaded_vvres != NULL:
        for i in range(num_threads):
            if threaded_vvres[i] != NULL:
                del threaded_vvres[i][0]
            PyMem_Free(threaded_vvres[i])
        PyMem_Free(threaded_vvres)