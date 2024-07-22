# cython: language_level=3
# distutils: language = c++

import numpy as np
import multiprocessing

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, free # Is needed to ensure thread-safety.
from libc.string cimport memset
cimport openmp
cimport cython
from cython.parallel import prange
from libc.math cimport fabs, pow, log, sqrt
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap
from libcpp.string cimport string
from libc.stdint cimport uint32_t, uint8_t, int64_t

from .ckdtree.ckdtree cimport cKDTree, query_ball_point
from . cimport utils
from .constants import FEATURE_NAMES

cdef double INFINITY
INFINITY = np.inf


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def compute_features(
    double [:, ::1] points,
    double search_radius,
    double max_graph_edge_length=INFINITY,
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
        int number_of_neighbors

        float [:, :] features = np.full((n_points, len(feature_names)), float("NaN"), dtype=np.float32)

        const np.float64_t[:, ::1] radius_vector
        np.float64_t p = <np.float64_t> float(2 if euclidean_distance else 1)
        np.float64_t eps_scipy = <np.float64_t> float(0)
        vector[np.intp_t] *** threaded_vvres
        int return_length = int(False)

        bint compute_graph_distance
        unsigned int start_node_id, max_edge_weight_count, neighbor_point_id_offset, row, column, coordinate_index, edge_weight_id, threshold_length, path_index
        double edge_weight
        double [:, ::1] edge_weights, edge_vector, shortest_edge_weights, new_array
        unsigned int [:, ::1] over_threshold_indices

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

    compute_graph_distance = <bint> (max_graph_edge_length < search_radius)

    if compute_graph_distance:
        # max edge weight count equals (n²-n)/2 (similar to nth triangular number), where n is the number of points, because
        # n² : every node has an edge with every node
        # -n : edge between point a-a, b-b et cetera is always zero and redundant
        # /2 : only one weight per edge, undirected graph a->b == b<-a
        max_edge_weight_count = (max_k_neighbors * max_k_neighbors - max_k_neighbors) // 2

        edge_weights = np.empty((num_threads, max_edge_weight_count), dtype=np.float64)
        edge_vector = np.empty((num_threads, 3), dtype=np.float64)
        shortest_edge_weights = np.empty((num_threads, max_k_neighbors), dtype=np.float64)
        over_threshold_indices = np.empty((num_threads, max_k_neighbors), dtype=np.uint32)
        new_array = np.empty((num_threads, max_k_neighbors), dtype=np.float64) # TODO Find better name

    threaded_vvres = init_result_vectors(num_threads)

    try:
        for i in prange(n_points, nogil=True, num_threads=num_threads):
            thread_id = openmp.omp_get_thread_num()
            neighbor_point_id_offset = thread_id * max_k_neighbors

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
            number_of_neighbors = n_neighbors_at_id

            if n_neighbors_at_id > max_k_neighbors:
                n_neighbors_at_id = max_k_neighbors
            elif n_neighbors_at_id == 0:
                continue

            for j in range(n_neighbors_at_id):
                neighbor_id = threaded_vvres[thread_id][0][0][j]
                for k in range(3):
                    neighbor_points[k, neighbor_point_id_offset + j] = kdtree.cself.raw_data[neighbor_id * 3 + k]

            if compute_graph_distance:
                for row in range(n_neighbors_at_id):
                    for column in range(n_neighbors_at_id):

                        if column >= row:
                            continue

                        for coordinate_index in range(3):
                            edge_vector[thread_id, coordinate_index] = (neighbor_points[coordinate_index, neighbor_point_id_offset + row]
                                                                        - neighbor_points[coordinate_index, neighbor_point_id_offset + column])

                        if euclidean_distance:
                            edge_weight = sqrt(edge_vector[thread_id, 0] * edge_vector[thread_id, 0]
                                                + edge_vector[thread_id, 1] * edge_vector[thread_id, 1]
                                                + edge_vector[thread_id, 2] * edge_vector[thread_id, 2])
                        else: # Manhattan distance.
                            edge_weight = (fabs(edge_vector[thread_id, 0])
                                            + fabs(edge_vector[thread_id, 1])
                                            + fabs(edge_vector[thread_id, 2]))

                        if edge_weight > max_graph_edge_length:
                            edge_weight = INFINITY

                        edge_weight_id = (row * (row + 1)) // 2 + (column - row) # sum of an arithmetic series        
                        edge_weights[thread_id, edge_weight_id] = edge_weight

                # TODO This is weird. Would be great if [0] is the start node in the first place.
                for start_node_id in range(n_neighbors_at_id):
                    if (neighbor_points[0, neighbor_point_id_offset + start_node_id] == points[i, 0] and
                        neighbor_points[1, neighbor_point_id_offset + start_node_id] == points[i, 1] and
                        neighbor_points[2, neighbor_point_id_offset + start_node_id] == points[i, 2]):
                        break

                shortest_edge_weights[thread_id, :n_neighbors_at_id] = dijkstra_all_shortest_edge_paths_weights(start_node_id, edge_weights[thread_id, :], n_neighbors_at_id)

                threshold_length = 0
                for path_index in range(n_neighbors_at_id):
                    if shortest_edge_weights[thread_id, path_index] > search_radius:
                        over_threshold_indices[thread_id, threshold_length] = path_index
                        threshold_length = threshold_length + 1

                n_neighbors_at_id = n_neighbors_at_id - threshold_length
                number_of_neighbors = number_of_neighbors - threshold_length

                for k in range(3):
                    neighbor_points[k, neighbor_point_id_offset : neighbor_point_id_offset + n_neighbors_at_id] = move_to_end(
                        new_array[thread_id, :n_neighbors_at_id],
                        neighbor_points[k, neighbor_point_id_offset : neighbor_point_id_offset + n_neighbors_at_id + threshold_length],
                        over_threshold_indices[thread_id, :threshold_length])

            utils.c_covariance(
                neighbor_points[:, neighbor_point_id_offset : neighbor_point_id_offset + n_neighbors_at_id],
                eigenvectors[:, thread_id * 3:(thread_id + 1) * 3],
            )
            utils.c_eigenvectors(
                eigenvectors[:, thread_id * 3:(thread_id + 1) * 3],
                eigenvalues[thread_id * 3:(thread_id + 1) * 3],
            )

            compute_features_from_eigenvectors(
                number_of_neighbors,
                eigenvalues[thread_id * 3 : thread_id * 3 + 3],
                eigenvectors[:, thread_id * 3 : thread_id * 3 + 3],
                features[i, :],
                features_map,
            )

    finally:
        free_result_vectors(threaded_vvres, num_threads)

    return np.array(features)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline double[:] move_to_end(double[:] new_array, double[:] array, unsigned int[:] indices_to_move) nogil:
    cdef bint move
    cdef unsigned int array_length, indices_length, new_array_index
    array_length = array.shape[0]
    indices_length = indices_to_move.shape[0]
    new_array_index = 0

    for array_index in range(array_length):

        move = <bint> False
        for indices_index in range(indices_length):
            if array_index == indices_to_move[indices_index]:
                move = <bint> True
                break

        if not move:
            new_array[new_array_index] = array[array_index]
            new_array_index = new_array_index + 1

    return new_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline double[:] dijkstra_all_shortest_edge_paths_weights(unsigned int start_node_id, double[:] edge_weights, unsigned int node_count) nogil:
    cdef double* shortest_edges_weights
    shortest_edges_weights = <double*> malloc(<size_t>(node_count * sizeof(double)))
    cdef bint* queue
    queue = <bint*> malloc(<size_t>(node_count * sizeof(bint)))
    cdef unsigned int queue_size
    queue_size = node_count

    cdef double candidate_weight, candidate_distance
    cdef unsigned int node_id, queue_node_id, candidate_node_id, row, column, edge_weight_id, row_index

    for node_id in range(node_count):
        shortest_edges_weights[node_id] = INFINITY
        queue[node_id] = True
    shortest_edges_weights[start_node_id] = 0

    while queue_size > 0:

        min_distance = INFINITY
        for node_id in range(node_count):
            candidate_distance = shortest_edges_weights[node_id]
            if queue[node_id] and candidate_distance <= min_distance:
                queue_node_id = node_id
                min_distance = candidate_distance
            
        queue[queue_node_id] = False
        queue_size -= 1

        for candidate_node_id in range(node_count):
            if queue[candidate_node_id] and queue_node_id != candidate_node_id:

                row = max(candidate_node_id, queue_node_id)
                column = min(candidate_node_id, queue_node_id)
                edge_weight_id = (row * (row + 1)) // 2 + (column - row) # sum of an arithmetic series

                candidate_weight = shortest_edges_weights[queue_node_id] + edge_weights[edge_weight_id]
                if candidate_weight < shortest_edges_weights[candidate_node_id]:
                    shortest_edges_weights[candidate_node_id] = candidate_weight

    free(queue)
 
    # Reusing the allocated memory.
    # TODO Check whether this is a good 'Cythonic' practice.
    for i in range(node_count):
        edge_weights[i] = shortest_edges_weights[i]
    free(shortest_edges_weights)
    return edge_weights[:node_count]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void compute_features_from_eigenvectors(
    int number_of_neighbors,
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

    if out_map.count(b"eigenvalue1"):
        out[out_map.at(b"eigenvalue1")] = l1
    if out_map.count(b"eigenvalue2"):
        out[out_map.at(b"eigenvalue2")] = l2
    if out_map.count(b"eigenvalue3"):
        out[out_map.at(b"eigenvalue3")] = l3

    if out_map.count(b"number_of_neighbors"):
        out[out_map.at(b"number_of_neighbors")] = number_of_neighbors

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

    if out_map.count(b"eigenvector1x"):
        out[out_map.at(b"eigenvector1x")] = eigenvectors[0, 0]
    if out_map.count(b"eigenvector1y"):
        out[out_map.at(b"eigenvector1y")] = eigenvectors[0, 1]
    if out_map.count(b"eigenvector1z"):
        out[out_map.at(b"eigenvector1z")] = eigenvectors[0, 2]

    if out_map.count(b"eigenvector2x"):
        out[out_map.at(b"eigenvector2x")] = eigenvectors[1, 0]
    if out_map.count(b"eigenvector2y"):
        out[out_map.at(b"eigenvector2y")] = eigenvectors[1, 1]
    if out_map.count(b"eigenvector2z"):
        out[out_map.at(b"eigenvector2z")] = eigenvectors[1, 2]

    if out_map.count(b"eigenvector3x"):
        out[out_map.at(b"eigenvector3x")] = eigenvectors[2, 0]
    if out_map.count(b"eigenvector3y"):
        out[out_map.at(b"eigenvector3y")] = eigenvectors[2, 1]
    if out_map.count(b"eigenvector3z"):
        out[out_map.at(b"eigenvector3z")] = eigenvectors[2, 2]


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
        
