from pathlib import Path

import laspy
import numpy as np
import pytest

import jakteristics
from jakteristics import FEATURE_NAMES, extension, las_utils, utils


data_dir = Path(__file__).parent / "data"


def test_matmul_transposed():
    points = np.random.rand(3, 4).astype("d")
    np_dot = np.dot(points, points.T)

    result = utils.py_matmul_transposed(points)

    assert np.allclose(np_dot, result)


def test_substract_mean():
    points = np.random.rand(3, 4).astype("d")
    expected = points - points.mean(axis=1)[:, None]
    result = np.asfortranarray(points.copy())
    utils.substract_mean(result)

    assert np.allclose(expected, result)


def test_covariance():
    points = np.random.rand(3, 4).astype("d")
    np_cov = np.cov(points)

    cov = utils.py_covariance(points)

    assert np.allclose(np_cov, cov)


def test_eigenvalues():
    """ write some tests """
    # --- given ---
    points = np.random.rand(3, 4).astype("d")
    np_cov = np.asfortranarray(np.cov(points).astype("d"))

    np_eigenvalues, np_eigenvectors = np.linalg.eig(np_cov)
    np_eigenvalues = np.abs(np_eigenvalues)

    # reorder eigenvectors before comparison
    argsort = list(reversed(np.argsort(np_eigenvalues)))
    np_eigenvectors = np.array([np_eigenvectors[:, i] for i in argsort])

    # --- when ---
    eigenvalues, eigenvectors = utils.py_eigenvectors(np_cov)

    # flip eigenvectors that are in the opposite direction (for comparison)
    for i in range(3):
        same_sign = (
            eigenvectors[i, 0] < 0
            and np_eigenvectors[i, 0] < 0
            or eigenvectors[i, 0] > 0
            and np_eigenvectors[i, 0] > 0
        )
        if not same_sign:
            np_eigenvectors[i, :] *= -1

    # --- then ---
    assert np.allclose(eigenvalues, np_eigenvalues[argsort])

    assert np.allclose(np_eigenvectors, eigenvectors, atol=1e-4)


def test_compute_features():
    n_points = 1000
    points = np.random.random((n_points, 3)) * 10

    features = extension.compute_features(points, 0.15)

    assert features.shape == (n_points, 14)


def test_compute_some_features():
    input_path = data_dir / "test_0.02_seconde.las"
    xyz = las_utils.read_las_xyz(input_path)
    n_points = xyz.shape[0]
    all_features = extension.compute_features(xyz, 0.15)

    for name in FEATURE_NAMES:
        features = extension.compute_features(xyz, 0.15, feature_names=[name])
        index = FEATURE_NAMES.index(name)

        assert features.shape == (n_points, 1)
        assert np.allclose(all_features[:, index], features.reshape(-1), equal_nan=True)


def test_write_extra_dims(tmp_path):
    input_path = data_dir / "test_0.02_seconde.las"
    output_path = tmp_path / "test_output.las"

    xyz = las_utils.read_las_xyz(input_path)

    features = extension.compute_features(xyz, 0.15)

    las_utils.write_with_extra_dims(input_path, output_path, features, FEATURE_NAMES)

    output_features = []
    with laspy.open(output_path, mode="r") as las:
        las_data = las.read()
        xyz_out = las_data.xyz
        for spec in las.header.point_format.extra_dimensions:
            name = spec.name.encode().replace(b"\x00", b"").decode()
            output_features.append(getattr(las_data, name))

        output_features = np.vstack(output_features).T

    assert np.allclose(xyz, xyz_out)
    assert np.allclose(features, output_features, equal_nan=True)


def test_not_contiguous():
    points = np.random.random((3, 1000)).T

    features = jakteristics.compute_features(points, 0.15)

    assert features.shape == (1000, 14)


def test_wrong_shape():
    points = np.random.random((3, 1000))

    with pytest.raises(ValueError):
        extension.compute_features(points, 0.15)


def test_nan():
    points = np.random.random((3, 1000)).T

    # compute kdtree where points are not located
    kdtree = jakteristics.cKDTree(points + 2)

    features = jakteristics.compute_features(points, 0.15, kdtree=kdtree)
    assert np.all(np.isnan(features))
