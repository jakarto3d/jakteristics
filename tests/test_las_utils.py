from pathlib import Path

import numpy as np
import pytest

from jakteristics import las_utils


TEST_DATA = Path(__file__).parent / "data"


def test_read_las():
    input_file = TEST_DATA / "test_0.02_seconde.las"
    xyz = las_utils.read_las_xyz(input_file)
    assert xyz.shape == (10310, 3)
    assert xyz.dtype == np.float64


def test_read_las_offset():
    input_file = TEST_DATA / "test_0.02_seconde.las"
    xyz, offset = las_utils.read_las_xyz(input_file, with_offset=True)
    assert xyz.shape == (10310, 3)
    assert xyz.dtype == np.float32
    assert np.allclose(offset, np.array([362327.0, 5157620.0, 106.271]))


def test_write_extra_dims_same_path():
    input_file = TEST_DATA / "test_0.02_seconde.las"
    with pytest.raises(ValueError):
        las_utils.write_with_extra_dims(
            input_file, input_file, np.array([1, 2, 3]), ["test_dim"]
        )


def test_write_extra_dims_wrong_point_count(temp_dir):
    input_file = TEST_DATA / "test_0.02_seconde.las"
    output_file = temp_dir / "out.las"
    with pytest.raises(ValueError) as e:
        las_utils.write_with_extra_dims(
            input_file, output_file, np.array([1, 2, 3]).T, ["test_dim"]
        )
