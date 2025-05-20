import json
import os

import numpy as np
import pytest

from oneqmc import convert_geo

INPUT_FILES = {
    "yaml": "../data/cyclobutadiene/square.yaml",
    "xyz": "test_square.xyz",
    "json": "test_square.json",
}


@pytest.mark.parametrize("from_fmt,to_fmt", [["yaml", "xyz"], ["yaml", "json"], ["json", "xyz"]])
def test_convert_geo(tmp_path, from_fmt, to_fmt):
    # define input and output files
    my_path = os.path.abspath(os.path.dirname(__file__))
    input = os.path.join(my_path, INPUT_FILES[from_fmt])
    output = os.path.join(tmp_path, f"tmp_square.{to_fmt}")
    charge = 0
    spin = 0
    unit = "angstrom"
    # the function can directly detect the file type by the file extension
    convert_geo.convert(input, output, charge, spin, unit)
    if to_fmt == "xyz":
        atoms1, coordinates1, extras1 = convert_geo.read_xyz(
            os.path.join(my_path, f"test_square.{to_fmt}")
        )
        atoms2, coordinates2, extras2 = convert_geo.read_xyz(
            os.path.join(tmp_path, f"tmp_square.{to_fmt}")
        )
        assert atoms1 == atoms2 and np.allclose(coordinates1, coordinates2) and extras1 == extras2
    else:
        with open(os.path.join(my_path, f"test_square.{to_fmt}")) as f:
            json1 = json.load(f)
        with open(os.path.join(tmp_path, f"tmp_square.{to_fmt}")) as f:
            json2 = json.load(f)
        assert match_dicts(json1, json2, ignore_keys=("provenance", "extras"))


def match_dicts(d1, d2, ignore_keys=()):
    """Compare two dictionaries recursively, ignoring some keys"""
    for k, v in d1.items():
        if k in ignore_keys:
            continue
        if isinstance(v, dict):
            return match_dicts(v, d2[k])
        else:
            if v != d2[k]:
                return False
    return True
