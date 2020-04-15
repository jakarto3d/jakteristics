
from libcpp.vector cimport vector
cimport numpy as np

cdef extern from "ckdtree_decl.h":
    struct ckdtreenode:
        np.intp_t split_dim
        np.intp_t children
        np.float64_t split
        np.intp_t start_idx
        np.intp_t end_idx
        ckdtreenode *less
        ckdtreenode *greater
        np.intp_t _less
        np.intp_t _greater

    struct ckdtree:
        vector[ckdtreenode]  *tree_buffer
        ckdtreenode   *ctree
        np.float64_t   *raw_data
        np.intp_t      n
        np.intp_t      m
        np.intp_t      leafsize
        np.float64_t   *raw_maxes
        np.float64_t   *raw_mins
        np.intp_t      *raw_indices
        np.float64_t   *raw_boxsize_data
        np.intp_t size

    int query_ball_point(const ckdtree *tree,
                            const np.float64_t *x,
                            const np.float64_t *r,
                            const np.float64_t p,
                            const np.float64_t eps,
                            const np.intp_t n_queries,
                            vector[np.intp_t] **results,
                            const int return_length) nogil except +

cdef class cKDTreeNode:
    cdef:
        readonly np.intp_t    level
        readonly np.intp_t    split_dim
        readonly np.intp_t    children
        readonly np.float64_t split
        ckdtreenode           *_node
        np.ndarray            _data
        np.ndarray            _indices
    cdef void _setup(cKDTreeNode self)

cdef class cKDTree:
    cdef:
        ckdtree * cself
        readonly cKDTreeNode     tree
        readonly np.ndarray      data
        readonly np.ndarray      maxes
        readonly np.ndarray      mins
        readonly np.ndarray      indices
        readonly object          boxsize
        np.ndarray               boxsize_data

    cdef _pre_init(cKDTree self)
    cdef _post_init(cKDTree self)
    cdef _post_init_traverse(cKDTree self, ckdtreenode *node)
