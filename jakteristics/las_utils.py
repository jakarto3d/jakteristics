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
    with laspy.file.File(filename, mode="r") as las:
        if with_offset:
            offset = las.header.offset
            out = np.empty((las.header.count, 3), dtype="f")
            points = np.stack(
                [las.x - offset[0], las.y - offset[1], las.z - offset[2]],
                axis=1,
                out=out,
            )
            return points, offset
        else:
            points = np.stack([las.x, las.y, las.z], axis=1)
            return points


def write_with_extra_dims(
    input_path: Path, output_path: Path, extra_dims: np.array, extra_dims_names: List,
):
    """From an existing las file, create a new las file with extra dimensions"""
    if input_path == output_path:
        raise ValueError("Paths must not be the same")

    header = laspy.file.File(input_path, mode="r").header

    if extra_dims.shape[0] != header.count:
        raise ValueError(
            "The features and point counts should be equal "
            "{extra_dims.shape[0]} != {header.count}"
        )
    with laspy.file.File(input_path, mode="r") as in_las:
        with laspy.file.File(output_path, mode="w", header=in_las.header) as out_las:
            data = [(name, extra_dims[:, i]) for i, name in enumerate(extra_dims_names)]

            for name, array in data:
                data_type = _get_las_data_type(array)
                out_las.define_new_dimension(name, data_type, name)

            for spec in in_las.reader.point_format:
                in_spec = in_las.reader.get_dimension(spec.name)
                out_las.writer.set_dimension(spec.name, in_spec)

            for name, array in data:
                setattr(out_las, name, array)


def _get_las_data_type(array):
    las_data_types = [
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
        # "S",  # strings not implemented
    ]
    type_ = str(array.dtype)
    if type_ not in las_data_types:
        raise NotImplementedError(f"Array type not implemented: {type_}")
    return las_data_types.index(type_) + 1
