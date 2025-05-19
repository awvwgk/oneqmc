import logging

from pyscf import gto
from pyscf.scf import RHF

log = logging.getLogger(__name__)


def pyscf_from_mol(mol, charge, spin, basis, cartesian=True, **kwargs):
    r"""Create a pyscf molecule and perform an SCF calculation on it.

    Args:
        mol (List[Tuple[int, np.ndarray]]): the molecule on which to perform the SCF calculation.
        charge (int): total charge to assign to the molecule.
        spin (int): total spin to assign to the molecule.
        basis (str): the name of the Gaussian basis set to use.
        cartesian (bool): Optional, whether to use cartesian or spherical Gaussian type orbitals.

    Returns:
        tuple: the pyscf molecule, the coefficients and the overlap values computed from SCF.
    """
    mol = gto.M(
        atom=mol,
        unit="bohr",
        basis=basis,
        charge=charge,
        spin=spin,
        cart=cartesian,
        parse_arg=False,
        verbose=0,
        **kwargs,
    )
    log.info("Running HF...")
    mf = RHF(mol)
    mf.kernel()
    log.info(f"HF energy: {mf.e_tot}")
    coeff = mf.mo_coeff
    overlap = mf.mol.intor("int1e_ovlp_cart" if cartesian else "int1e_ovlp")
    return mol, coeff, overlap
