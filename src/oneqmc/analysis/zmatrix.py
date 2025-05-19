from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from typing_extensions import Self

from ..molecule import Molecule

ZMatrixLine: TypeAlias = tuple[tuple[int, float], ...]


@dataclass
class ZMatrix:
    total_charge: int
    total_spin: int
    charges: Sequence[int]
    lines: Sequence[ZMatrixLine]

    def __post_init__(self):
        assert len(self.charges) == len(self.lines)

    def replace_single_value(self, line_idx: int, entry_idx: int, value: float) -> Self:
        assert entry_idx < min(line_idx, 3)
        new_lines = [
            (*line[:entry_idx], (line[entry_idx][0], value), *line[entry_idx + 1 :])
            if i == line_idx
            else line
            for i, line in enumerate(self.lines)
        ]
        return replace(self, lines=new_lines)

    def break_bond(
        self, bond_idx: int, max_distance: float = 3.0, n_structures: int = 5
    ) -> list[Self]:
        assert bond_idx > 0
        bond_lengths = np.linspace(self.lines[bond_idx][0][1], max_distance, n_structures)
        zmatrices = [
            self.replace_single_value(bond_idx, 0, bond_length.item())
            for bond_length in bond_lengths
        ]
        return zmatrices

    def distort_angle(
        self,
        angle_idx: int,
        delta: float = 0.5,
        n_structures: int = 5,
        min_value: float = 0.0,
        max_value: float = np.pi,
    ):
        assert angle_idx > 1
        assert min_value >= 0.0
        assert max_value <= np.pi
        current_value = self.lines[angle_idx][1][1]
        angles = np.linspace(
            max(min_value, current_value - delta),
            min(max_value, current_value + delta),
            n_structures,
        )
        return [self.replace_single_value(angle_idx, 1, angle.item()) for angle in angles]

    def distort_dihedral(
        self,
        dihedral_idx: int,
        delta: float = 0.5,
        n_structures: int = 5,
        min_value: float = -np.pi,
        max_value: float = np.pi,
    ):
        assert dihedral_idx > 2
        assert min_value >= -np.pi
        assert max_value <= np.pi
        current_value = self.lines[dihedral_idx][2][1]
        dihedrals = np.linspace(
            max(min_value, current_value - delta),
            min(max_value, current_value + delta),
            n_structures,
        )
        return [
            self.replace_single_value(dihedral_idx, 1, dihedral.item()) for dihedral in dihedrals
        ]

    def to_cartesian(self) -> np.ndarray:
        cartesian = np.zeros((0, 3))
        for i, zmatrix_line in enumerate(self.lines):
            cartesian = place_next_atom_of_zmatrix(cartesian, zmatrix_line)
            if not np.isfinite(cartesian).all():
                raise ValueError(f"Encountered a nan line {i}, {cartesian}")
        return cartesian

    def to_molecule(self, *, unit) -> Molecule:
        coords = self.to_cartesian()
        return Molecule.make(
            coords=np.array(coords),
            charges=np.array(self.charges),
            charge=self.total_charge,
            spin=self.total_spin,
            unit=unit,
        )


def direction_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    difference = x - y
    return difference / np.linalg.norm(difference)


def normed_cross_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cross = np.cross(x, y)
    return cross / np.linalg.norm(cross)


def rot_y(theta):
    """Return the rotation matrix about y-axis by angle theta."""
    return np.array(
        [
            [np.cos(theta), np.zeros_like(theta), np.sin(theta)],
            [np.zeros_like(theta), np.ones_like(theta), np.zeros_like(theta)],
            [-np.sin(theta), np.zeros_like(theta), np.cos(theta)],
        ]
    )


def place_next_atom_of_zmatrix(
    previous_cartesian: np.ndarray, zmatrix_line: ZMatrixLine
) -> np.ndarray:
    r"""Place the next atom according to the Z matrix line."""
    assert len(zmatrix_line) < 4
    if len(zmatrix_line) == 0:
        assert len(previous_cartesian) == 0
        return np.zeros((1, 3))
    bond_atom_idx, bond_value = zmatrix_line[0]
    if len(zmatrix_line) == 1:
        assert len(previous_cartesian) == 1
        assert bond_atom_idx == 0
        return np.concatenate(
            [
                previous_cartesian,
                previous_cartesian[bond_atom_idx][None] + np.array([bond_value, 0, 0]),
            ],
            axis=0,
        )
    angle_atom_idx, angle_value = zmatrix_line[1]
    if len(zmatrix_line) == 2:
        assert len(previous_cartesian) == 2
        assert bond_atom_idx < 2
        if bond_atom_idx == 0:
            r = np.array([bond_value, 0, 0])
        else:
            r = -np.array([bond_value, 0, 0])
        rotated_r = np.einsum("ij,j->i", rot_y(np.array(angle_value)), r)
        return np.concatenate(
            [
                previous_cartesian,
                rotated_r[None] + previous_cartesian[bond_atom_idx],
            ],
            axis=0,
        )
    dihedral_atom_idx, dihedral_value = zmatrix_line[2]
    r_cos_angle = np.cos(np.pi - angle_value) * bond_value
    r_sin_angle = np.sin(np.pi - angle_value) * bond_value
    bonded_atom_coord = previous_cartesian[bond_atom_idx]
    angle_atom_coord = previous_cartesian[angle_atom_idx]
    dihedral_atom_coord = previous_cartesian[dihedral_atom_idx]
    r = np.stack(
        [
            r_cos_angle,
            np.cos(dihedral_value) * r_sin_angle,
            np.sin(dihedral_value) * r_sin_angle,
        ]
    )
    BC = direction_vector(bonded_atom_coord, angle_atom_coord)
    AB = direction_vector(angle_atom_coord, dihedral_atom_coord)
    N = normed_cross_product(AB, BC)
    M = normed_cross_product(N, BC)
    rot = np.stack([BC, M, N], axis=1)
    r_final = bonded_atom_coord + np.dot(rot, r)
    return np.concatenate([previous_cartesian, r_final[None]], axis=0)
