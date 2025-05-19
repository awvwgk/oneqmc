import json
import logging
import os
import re
from typing import List, Optional, Sequence

import numpy as np
import qcelemental as qcel
import yaml
from tqdm import tqdm

from .molecule import ANGSTROM, Molecule
from .types import Embeddings

# Note, spin= 2*s (same as pyscf), and molecular multiplicity= 2*s+1
log = logging.getLogger("oneqmc")


def read_xyz(file_name):
    atoms = []
    coordinates = []

    with open(file_name) as f:
        data = f.readlines()
        extras = data[1].strip()
        lines = data[2:]  # Skip the first two lines
        for line in lines:
            tokens = line.strip().split()
            atom = tokens[0]
            coords = [float(token) for token in tokens[1:]]
            atoms.append(atom)
            coordinates.append(coords)
    return atoms, coordinates, extras


def xyz_string(atoms, coordinates, charge, spin, extras) -> str:
    if extras is not None:
        extras = " ".join([f"{key}: {value}" for key, value in extras.items()])
    return "\n".join(
        [
            str(len(atoms)),
            f"Charge: {charge}, Spin: {spin}, Extras: {extras}",
        ]
        + [f"{atom} {coord[0]} {coord[1]} {coord[2]}" for atom, coord in zip(atoms, coordinates)]
    )


def write_xyz(atoms, coordinates, charge, spin, extras, output_file):
    with open(output_file, "w") as f:
        f.write(xyz_string(atoms, coordinates, charge, spin, extras))


def read_yaml(file_name):
    with open(file_name, "r") as f:
        data = yaml.safe_load(f)
    atoms = [qcel.periodictable.to_symbol(i) for i in data["charges"]]
    coordinates = data["coords"]
    charge = float(data["charge"])
    spin = int(data["spin"])
    unit = data["unit"]
    extras = data.get("extras")
    return atoms, coordinates, charge, spin, extras, unit


def write_yaml(atoms, coordinates, output_file, charge=0.0, spin=0, extras=None, unit="angstrom"):
    charges = [qcel.periodictable.to_atomic_number(i) for i in atoms]
    if unit == "bohr":
        coordinates = [[i[0] * ANGSTROM, i[1] * ANGSTROM, i[2] * ANGSTROM] for i in coordinates]
    else:
        coordinates = [[i[0], i[1], i[2]] for i in coordinates]
    data = {
        "coords": coordinates,
        "charges": charges,
        "charge": charge,
        "spin": spin,
        "unit": unit,
        "extras": extras,
    }
    with open(output_file, "w") as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)


def convert(input, output, charge=0.0, spin=0, unit="angstrom"):
    # We could add more different geometry versions.
    input_extension = input.split(".")[-1].lower()
    output_extension = output.split(".")[-1].lower()
    if input_extension == "xyz":
        atoms, coordinates, extras = read_xyz(input)
    elif input_extension == "yaml":
        atoms, coordinates, charge, spin, extras, unit = read_yaml(input)
    elif input_extension == "json":
        molecule = qcel.models.Molecule.from_file(input).dict()
        atoms = molecule["symbols"].tolist()
        coordinates = molecule["geometry"].tolist()
        charge = molecule["molecular_charge"]
        spin = molecule["molecular_multiplicity"] - 1
        extras = molecule["extras"]["notes"]
        unit = "bohr"
    else:
        raise ValueError(f"Unsupported conversion: {input_extension} input")

    if output_extension == "xyz":
        if unit == "bohr":
            coordinates = [
                [
                    i[0] / ANGSTROM,
                    i[1] / ANGSTROM,
                    i[2] / ANGSTROM,
                ]
                for i in coordinates
            ]
        write_xyz(atoms, coordinates, charge, spin, extras, output)
    elif output_extension == "yaml":
        write_yaml(atoms, coordinates, output, charge=charge, spin=spin, extras=extras, unit=unit)
    elif output_extension == "json":
        # The default unit is Bohr in qcelemental
        if unit == "angstrom":
            coordinates = [[i[0] * ANGSTROM, i[1] * ANGSTROM, i[2] * ANGSTROM] for i in coordinates]
        molecule = qcel.models.Molecule(
            symbols=np.array(atoms),
            geometry=np.array(coordinates),
            molecular_charge=float(charge),
            molecular_multiplicity=int(spin + 1),
            extras={"notes": extras},
        )
        molecule.to_file(output)
    else:
        raise ValueError(f"Unsupported conversion: {output_extension} output")


def get_filenames_recursive(datadir, extensions: Optional[List[str]] = None):
    """Retrieve path to all files in a directory recursively, with optional extension filtering."""
    filenames = []
    for root, dirs, files in os.walk(datadir):
        for file in files:
            if extensions is None or np.any([file.endswith(ext) for ext in extensions]):
                filenames.append(os.path.join(root, file))
    return filenames


def load_molecules(
    datadir,
    file_whitelist: str | re.Pattern | None = None,
    json_whitelist: str | re.Pattern | None = None,
) -> Sequence[Molecule]:
    molecules = []
    filenames = sorted(get_filenames_recursive(datadir, extensions=["yaml", "json"]))
    for filename in tqdm(filenames, desc="Loading molecules"):
        if file_whitelist is None or re.match(file_whitelist, filename):
            input_extension = filename.split(".")[-1].lower()
            print(filename)
            with open(filename, mode="r") as stream:
                if input_extension == "yaml":
                    system_config = yaml.safe_load(stream)
                    if "data" in system_config.keys():
                        system_config["extras"].update(system_config.pop("data"))
                    if "embeddings" in system_config.keys():
                        embeddings_npz = np.load(os.path.join(datadir, system_config["embeddings"]))
                        embeddings = Embeddings.make(
                            embeddings_npz["vector"],
                            embeddings_npz["scalar"],
                        )
                        system_config["embeddings"] = embeddings
                    molecules.append(Molecule.from_dict(system_config))
                elif input_extension == "json":
                    data = json.load(stream)
                    for struct in tqdm(data.keys(), desc=f"Loading {filename}"):
                        if json_whitelist is None or re.match(json_whitelist, struct):
                            m = qcel.models.Molecule.from_data(data[struct])
                            molecules.append(Molecule.from_qcelemental(m))
                else:
                    raise TypeError(
                        f"Unsupported filetype: {input_extension} input is not supported"
                    )
    assert len(molecules) > 0, "Data missing. No matching molecules were found."
    log.info(f"Loaded dataset of {len(molecules)} molecules.")
    return molecules
