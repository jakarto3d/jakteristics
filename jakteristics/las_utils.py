from pathlib import Path
from typing import List, Tuple, Union


import numpy as np
import laspy


def read_las_xyz(
    filename: Union[str, Path], with_offset: bool = False
) -> Union[np.array, Tuple[np.ndarray, List[float]]]:
    """Reads xyz coordinates of a las file, optionally as single precision floating point.

    Arguments:
        filename:
            The las file to read
        with_offset:
            If True, returns a tuple of a float32 array of coordinates, and the las header offset
            If False, returns only float64 coordinates
            Default: False
    Returns:
        Depending on the `with_offset` parameter, either an (n x 3) array, or
        a tuple of an (n x 3) array and the file offset.
    """
    with laspy.open(filename, mode="r") as las:
        las_data = las.read()
        if with_offset:
            offset = las.header.offset
            points = np.ascontiguousarray(las_data.xyz - offset, dtype="f")
            return points, offset
        else:
            points = np.ascontiguousarray(las_data.xyz)
            return points


def write_with_extra_dims(
    input_path: Path,
    output_path: Path,
    extra_dims: np.array,
    extra_dims_names: List,
):
    """From an existing las file, create a new las file with extra dimensions

    Arguments:
        input_path: The input las file.
        output_path: The output las file.
        extra_dims: The numpy array containing geometric features.
        extra_dims_names: A list of names corresponding to each column of `extra_dims`.
    """
    if input_path == output_path:
        raise ValueError("Paths must not be the same")

    with laspy.open(input_path, mode="r") as in_las:
        header = in_las.header
        if extra_dims.shape[0] != header.point_count:
            raise ValueError(
                f"The features and point counts should be equal "
                f"{extra_dims.shape[0]} != {header.point_count}"
            )

        data = [(name, extra_dims[:, i]) for i, name in enumerate(extra_dims_names)]
        new_dimensions = [
            laspy.ExtraBytesParams(name=name, type=extra_dims.dtype, description=name)
            for name in extra_dims_names
        ]

        # insert new data in previous pointcloud PackedPointRecord
        las_data = in_las.read()
        las_data.add_extra_dims(new_dimensions)
        las_data.update_header()
        for name, array in data:
            setattr(las_data, name, array)

        with laspy.open(output_path, mode="w", header=las_data.header) as out_las:
            out_las.write_points(las_data.points)
