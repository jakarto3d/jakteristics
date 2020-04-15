from typing import List

import numpy as np

import jakteristics
from jakteristics.ckdtree import ckdtree

from .constants import FEATURE_NAMES


def compute_features(
    points: np.ndarray,
    search_radius: float,
    *,
    kdtree: ckdtree.cKDTree = None,
    num_threads: int = -1,
    max_k_neighbors: int = 50000,
    euclidean_distance: bool = True,
    feature_names: List[str] = FEATURE_NAMES,
    eps: float = 0.0,
) -> np.ndarray:
    """
    Compute features for a set of points.

    Parameters:
        points:
            A contiguous (n, 3) array of xyz coordinates to query.
        search_radius:
            The radius to query neighbors at each point.
        kdtree:
            If None, the kdtree is computed from the list of points.
            Must be an instance of `jakteristics.cKDTree`
            (and not `scipy.spatial.cKDTree`).
        num_threads:
            The number of threads (OpenMP) to use when doing the computation.
            Default: The number of cores on the machine.
        max_k_neighbors:
            The maximum number of neighbors to query
            Larger number will use more memory, but the neighbor points are not
            all kept at the same time in memory.
            Note: if this number is smaller, the neighbor search will not be faster.
            The radius is used to do the query, and the neighbors are then removed
            according to this parameter.
        euclidean_distance:
            How to compute the distance between 2 points.
            If true, the Euclidean distance is used.
            If false, the sum-of-absolute-values is used ("Manhattan" distance).
        feature_names:
            The feature names to compute (see `constants.FEATURE_NAMES` for possible values)
            Default: all features
        eps:
            Return approximate nearest neighbors; the k-th returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.

    Returns:
        The computed features, one row per query point, and one column 
        per requested feature.
    """

    points = np.ascontiguousarray(points)

    return jakteristics.extension.compute_features(
        points,
        search_radius,
        kdtree=kdtree,
        num_threads=num_threads,
        max_k_neighbors=max_k_neighbors,
        euclidean_distance=euclidean_distance,
        feature_names=feature_names,
        eps=eps,
    )
