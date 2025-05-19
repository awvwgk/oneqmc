import py3Dmol
import qcelemental as qcel

from ..convert_geo import xyz_string
from ..molecule import ANGSTROM, Molecule
from ..types import MolecularConfiguration


def mol_conf_to_xyz(mol_conf: MolecularConfiguration) -> str:
    assert mol_conf.nuclei.charges.ndim == 1
    mask = mol_conf.nuclei.mask
    return xyz_string(
        [qcel.periodictable.to_symbol(i) for i in mol_conf.nuclei.charges[mask]],
        mol_conf.nuclei.coords[mask] / ANGSTROM,
        mol_conf.total_charge.item(),
        mol_conf.total_spin.item(),
        {},
    )


def show_mol_conf(mol_conf: MolecularConfiguration, view=None):
    if view is None:
        view = py3Dmol.view()
    view.addModel(mol_conf_to_xyz(mol_conf))
    view.setStyle({"model": -1}, {"sphere": {"scale": 0.3}, "stick": {}})
    view.zoomTo()
    return view


def show_mol(mol: Molecule, view=None):
    mol_conf = mol.to_mol_conf(len(mol))
    return show_mol_conf(mol_conf, view=view)
