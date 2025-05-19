from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from itertools import count
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import qcelemental as qcel
import yaml

from .types import Embeddings, MolecularConfiguration, Nuclei

ANGSTROM = 1 / 0.52917721092

__all__ = ["Molecule"]


def get_shell(z):
    # returns the number of (at least partially) occupied shells for 'z' electrons
    # 'get_shell(z+1)-1' yields the number of fully occupied shells for 'z' electrons
    max_elec = 0
    n = 0
    for n in count():
        if z <= max_elec:
            break
        max_elec += 2 * (1 + n) ** 2
    return n


def parse_molecules():
    data = {}
    for t in resources.files("oneqmc").joinpath("conf/mol").iterdir():
        if t.is_file() and t.name.endswith(".yaml"):
            with t.open() as f:
                name = t.name.removesuffix(".yaml")
                data[name] = yaml.safe_load(f)
    return data


_SYSTEMS = parse_molecules()


@dataclass(frozen=True)
class Molecule:
    r"""Represents a molecule.

    The array-like arguments accept anything that can be transformed to
    :class:`jax.Array`.

    Args:
        coords (float, (:math:`N_\text{nuc}`, 3), a.u.):
            nuclear coordinates as rows
        charges (int, (:math:`N_\text{nuc}`)): atom charges
        charge (int): total charge of a molecule
        spin (int): total spin multiplicity
        extras (optional, dict): the extra information to be stored in the molecule object
    """

    all_names: ClassVar[set] = set(_SYSTEMS.keys())

    coords: np.ndarray
    charges: np.ndarray
    charge: int
    spin: int
    embeddings: Embeddings[np.ndarray] | None
    extras: dict

    # DERIVED PROPERTIES:
    n_nuc: int
    n_atom_types: int
    n_up: int
    n_down: int
    species: np.ndarray
    # total numbers of occupied shells
    n_shells: tuple

    @staticmethod
    def make(
        *,
        coords: np.ndarray,
        charges: np.ndarray,
        charge: int,
        spin: int,
        embeddings: Embeddings[np.ndarray] | None = None,
        unit="bohr",
        extras=None,
    ) -> Molecule:
        if unit == "angstrom":
            unit_multiplier = ANGSTROM
        elif unit == "bohr":
            unit_multiplier = 1.0
        else:
            raise ValueError(f"Unknown unit: {unit}")

        n_atom_types = len(np.unique(charges))
        species = charges

        n_elec = int(sum(charges) - charge)
        assert (
            not (n_elec + spin) % 2
        ), f"Molecule with charges {charges} has incompatible total spin"

        if (n_elec + spin) % 2:
            raise ValueError(f"Invalid number of electrons: {n_elec} + {spin}")

        # some paranoid checks
        n_nuc = int(len(coords))
        assert isinstance(coords, np.ndarray), type(coords)
        assert coords.shape == (n_nuc, 3), coords.shape
        assert isinstance(charges, np.ndarray), type(charges)
        assert charges.shape == (n_nuc,), charges.shape
        assert isinstance(charge, int), type(charge)
        assert isinstance(spin, int), type(spin)
        assert isinstance(extras, dict | None), type(extras)
        assert isinstance(unit, str), type(unit)

        return Molecule(
            coords=unit_multiplier * np.asarray(coords),
            charges=charges,
            charge=charge,
            spin=spin,
            extras=extras or {},
            embeddings=embeddings,
            species=species,
            n_nuc=len(charges),
            n_atom_types=n_atom_types,
            n_up=(n_elec + spin) // 2,
            n_down=(n_elec - spin) // 2,
            n_shells=tuple(get_shell(z) for z in charges),
        )

    def __len__(self):
        return len(self.charges)

    def __repr__(self):
        return (
            "Molecule(\n"
            f"  coords=\n{self.coords},\n"
            f"  charges={self.charges},\n"
            f"  charge={self.charge},\n"
            f"  spin={self.spin}\n"
            f"  n_nuc={self.n_nuc}\n"
            f"  n_up={self.n_up}\n"
            f"  n_down={self.n_down}\n"
            f"  species={self.species}\n"
            ")"
        )

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def as_pyscf(self):
        return [(int(charge), coord) for coord, charge in zip(self.coords, self.charges)]

    @staticmethod
    def from_name(name, **kwargs) -> Molecule:
        """Create a molecule from a database of named molecules.

        The available names are in :attr:`Molecule.all_names`.
        """
        if name in _SYSTEMS:
            system = deepcopy(_SYSTEMS[name])
            system.update(kwargs)
        else:
            raise ValueError(f"Unknown molecule name: {name}")
        return Molecule.from_dict(system)

    @staticmethod
    def from_dict(system: dict[str, Any]) -> Molecule:
        """Create a molecule from a Python dictionary."""
        return Molecule.make(
            coords=np.array(system["coords"]),
            charges=np.array(system["charges"]),
            charge=int(system["charge"]),
            spin=int(system["spin"]),
            unit=system.get("unit", "bohr"),
            extras=system.get("extras", {}),
        )

    @staticmethod
    def from_qcelemental(qcel_mol: qcel.models.Molecule) -> Molecule:
        """Create a molecule from a qcelemental molecule object.

        Accept a qcel.models.Molecule instance and return a oneqmc.molecule.Molecule
        instance of the same molecule.
        """
        coords = qcel_mol.geometry
        charges = [qcel.periodictable.to_atomic_number(i) for i in qcel_mol.symbols]
        charge = qcel_mol.molecular_charge
        spin = qcel_mol.molecular_multiplicity - 1
        unit = "bohr"
        return Molecule.make(
            coords=coords,
            charges=np.array(charges),
            charge=int(charge),
            spin=spin,
            unit=unit,
        )

    def to_qcelemental(self) -> qcel.models.Molecule:
        atoms = [qcel.periodictable.to_symbol(i) for i in self.species]
        extras: dict[str, Any] = {"notes": self.extras}
        return qcel.models.Molecule(
            symbols=np.array(atoms),
            geometry=np.array(self.coords),
            molecular_charge=float(self.charge),
            molecular_multiplicity=int(self.spin + 1),
            extras=extras,
        )

    def to_mol_conf(self, max_nuc):
        nucs = Nuclei.embed_to_masked(
            jax.device_put(self.coords),
            jax.device_put(self.charges),
            jax.device_put(self.species),
            jax.device_put(self.embeddings),
            max_nuc,
        )
        return MolecularConfiguration(nucs, jnp.asarray(self.charge), jnp.asarray(self.spin))
