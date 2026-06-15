# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import json
import logging
import os
from contextlib import nullcontext
from enum import Enum
from os import PathLike
from typing import IO, Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
import yaml  # type: ignore

logger = logging.getLogger("dinov3")


# This type represents a list of results, e.g. baselines for an evaluation.
Results: Any = pd.DataFrame

try:
    import openpyxl  # noqa: F401

    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    logger.warning("can't import openpyxl package")


PathOrFileLikeObject = Union[str, PathLike, IO]


class FileFormat(Enum):
    CSV = "csv"
    JSON_LINES = "json-lines"
    EXCEL = "excel"
    YAML = "yaml"

    @staticmethod
    def guess(path: Union[str, PathLike]) -> "FileFormat":
        _, ext = os.path.splitext(path)
        supported_exts = {
            ".csv": FileFormat.CSV,
            ".jsonl": FileFormat.JSON_LINES,
            ".excel": FileFormat.EXCEL,
            ".yaml": FileFormat.YAML,
        }
        if ext not in supported_exts:
            raise ValueError(f"Passed path has extension {ext}, only {list(supported_exts.keys())} are supported.")
        return supported_exts[ext]


_INT_DTYPES = [
    pd.Int8Dtype(),
    pd.Int16Dtype(),
    pd.Int32Dtype(),
    pd.UInt8Dtype(),
    pd.UInt16Dtype(),
    pd.UInt32Dtype(),
    pd.Int64Dtype(),
]

_FLOAT_DTYPES = [
    pd.Float32Dtype(),
    pd.Float64Dtype(),
]

_TO_STRING_DTYPES = [
    pd.BooleanDtype(),
]

_VALID_DTYPES = [
    pd.StringDtype(),
    pd.Int64Dtype(),
    pd.Float64Dtype(),
]


def _map_dtypes(results: Results) -> Results:
    results = results.convert_dtypes(
        infer_objects=True,
        convert_string=True,
        convert_integer=True,
        convert_boolean=True,
        convert_floating=True,
    )
    for column_name in results.columns:
        if results.dtypes[column_name] in _INT_DTYPES:
            results[column_name] = results[column_name].astype(pd.Int64Dtype())
        elif results.dtypes[column_name] in _FLOAT_DTYPES:
            results[column_name] = results[column_name].astype(pd.Float64Dtype())
        elif results.dtypes[column_name] in _TO_STRING_DTYPES:
            results[column_name] = results[column_name].astype(pd.StringDtype())

    return results


def _validate_column(results: Results, *, name: str, dtype: Union[str, type]) -> bool:
    try:
        loc = results.columns.get_loc(name)
    except KeyError:
        return False
    return results.dtypes[loc] == dtype


def _validate(results: Results) -> bool:
    for column_name in results.columns:
        dtype = results.dtypes[column_name]
        if dtype not in _VALID_DTYPES:
            return False

    return True


def _assert_valid_dtypes(results: Results) -> None:
    assert _validate(results), f"All dtypes from {results.dtypes} must be in {_VALID_DTYPES}"


Scalar = Union[str, int, float]


def _map_scalar(x: Scalar) -> List[Scalar]:
    return [x]


def _map_scalar_list(x: List[Scalar]) -> List[Scalar]:
    return x


def make(data: Dict[str, Union[str, int, float]]) -> Results:
    """Construct results from a dictionary of scalars or lists of scalars."""

    map_value: Callable[..., List[Scalar]]
    if all((isinstance(value, Sequence) for key, value in data.items())):
        map_value = _map_scalar_list
    else:
        map_value = _map_scalar
    results = pd.DataFrame({key: map_value(value) for key, value in data.items()})
    results = _map_dtypes(results)
    _assert_valid_dtypes(results)
    return results


def vstack(*results_sequence: Sequence[Results]) -> Results:
    """Concatenate (vertically) results."""

    return pd.concat(results_sequence, axis=0, ignore_index=True)


def load(f: PathOrFileLikeObject, file_format: Optional[FileFormat] = None) -> Results:
    """Load results from a file via a path-like object or from a file-like object."""

    if isinstance(f, (str, PathLike)):
        file_format = FileFormat.guess(f)
    elif file_format is None:
        raise ValueError("No file format specified for file-like object")

    assert file_format is not None
    if file_format == FileFormat.CSV:
        results = pd.read_csv(f, sep=",", na_values="", header=0)
    elif file_format == FileFormat.JSON_LINES:
        results = pd.read_json(f, lines=True)
    elif file_format == FileFormat.EXCEL:
        results = pd.read_excel(f)
    elif file_format == FileFormat.YAML:
        with open(f) as file:  # type: ignore
            results = pd.DataFrame.from_dict(yaml.safe_load(file), orient="index")
    else:
        raise ValueError("Unsupported file format: {file_format}")

    results = _map_dtypes(results)
    _assert_valid_dtypes(results)
    return results


def load_collection(f: PathOrFileLikeObject) -> Dict[str, Results]:
    """Load a collection of results from a file via a path-like object or from a file-like object."""

    results_collection = pd.read_excel(f, sheet_name=None)

    for sheet_name, results in results_collection.items():
        results = _map_dtypes(results)
        _assert_valid_dtypes(results)
        results_collection[sheet_name] = results
    return results_collection


def save(
    results: Sequence[Results],
    f: PathOrFileLikeObject,
    file_format: Optional[FileFormat] = None,
) -> None:
    """Save results to a file via a path-like object or to a file-like object."""

    _assert_valid_dtypes(results)

    if isinstance(f, (str, PathLike)):
        file_format = FileFormat.guess(f)
    elif file_format is None:
        raise ValueError("No file format specified for file-like object")

    assert file_format is not None
    if file_format == FileFormat.CSV:
        results.to_csv(f, index=False, header=True, sep=",", na_rep="")  # type: ignore
    elif file_format == FileFormat.JSON_LINES:
        # NOTE: pandas escapes '/' characters
        s = results.to_json(orient="records", lines=True, indent=None)  # type: ignore
        if isinstance(f, (str, PathLike)):
            context = open(f, "w")  # type: ignore
        else:
            context = nullcontext(enter_result=f)  # type: ignore
        with context as f:
            for line in s.splitlines():
                line = json.dumps(json.loads(line), separators=(",", ":"))
                f.write(line + "\n")
    elif file_format == FileFormat.EXCEL:
        results.to_excel(f, header=True, index=False, na_rep="")  # type: ignore
    elif file_format == FileFormat.YAML:
        with open(f, "w") as fp:  # type: ignore
            yaml.safe_dump(results.to_dict(orient="index"), fp, default_flow_style=False)  # type: ignore
    else:
        raise ValueError("Unsupported file format: {file_format}")


def save_from_dict(
    results_dict: Dict[str, Union[str, int, float]],
    results_path: PathOrFileLikeObject,
) -> None:
    results = make(results_dict)
    save(results, results_path)


def save_collection(
    results_collection: Dict[str, Results],
    f: PathOrFileLikeObject,
) -> None:
    """Save a collection of results to a file via a path-like object or to a file-like object."""

    if not HAS_OPENPYXL:
        logger.warning("openpyxl need to be installed, passing...")
        return

    with pd.ExcelWriter(f, engine="openpyxl", mode="w") as writer:
        for sheet_name, results in results_collection.items():
            _assert_valid_dtypes(results)
            results.to_excel(writer, sheet_name=sheet_name, header=True, index=False, na_rep="")
