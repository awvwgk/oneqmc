from typing import Sequence

import jax
import jax.numpy as jnp
from joblib import Parallel, delayed
from tqdm import tqdm

from ...geom import masked_pairwise_diffs
from ...molecule import Molecule
from ...types import ElectronConfiguration, ModelDimensions, MolecularConfiguration, Psi, ScfParams
from ...utils import zero_embed
from ..base import WaveFunction, eval_log_slater
from .gto import evaluate_gtos, gto_spec_from_pyscf
from .pyscfext import pyscf_from_mol


class HartreeFock(WaveFunction):
    r"""Hartree-Fock wavefunction used as a transferable baseline."""

    def __init__(self, dims: ModelDimensions):
        super().__init__(dims)

    def __call__(
        self, electrons: ElectronConfiguration, inputs, return_mos=False, return_det_dist=False
    ):
        mol_conf: MolecularConfiguration = inputs["mol"]
        (idxs, shells, mo_coeff_up, mo_coeff_down) = inputs["scf"]
        diffs, _ = masked_pairwise_diffs(
            electrons.coords, mol_conf.nuclei.coords, electrons.mask, mol_conf.nuclei.mask
        )
        aos = evaluate_gtos(diffs, shells, idxs)
        det_up = jnp.einsum("...mo,...me->...eo", mo_coeff_up, aos[..., : electrons.max_up])
        det_down = jnp.einsum("...mo,...me->...eo", mo_coeff_down, aos[..., electrons.max_up :])
        slater_mask_up = jnp.logical_and(electrons.up.mask[:, None], electrons.up.mask[None, :])
        slater_mask_down = jnp.logical_and(
            electrons.down.mask[:, None], electrons.down.mask[None, :]
        )
        if return_mos:
            # Add n_determinants axis
            return [
                x[..., None, :, :] for x in [det_up, det_down, slater_mask_up, slater_mask_down]
            ]
        det_up = jnp.where(slater_mask_up, det_up, jnp.eye(electrons.max_up))
        det_down = jnp.where(slater_mask_down, det_down, jnp.eye(electrons.max_down))
        sign_up, det_up = eval_log_slater(det_up)
        sign_down, det_down = eval_log_slater(det_down)
        psi = Psi(jax.lax.stop_gradient(sign_up * sign_down), det_up + det_down)
        if return_det_dist:
            return psi, jnp.array([0.0])  # Only one determinant here
        else:
            return psi

    @classmethod
    def from_mol(
        cls,
        mols: Sequence[Molecule],
        dims: ModelDimensions,
        *,
        basis: str = "6-31G",
        max_n_gaussians: int = 6,
        **pyscf_kwargs,
    ) -> ScfParams:
        r"""Create input to the constructor from a :class:`~oneqmc.molecule.Molecule`.

        Args:
            mols (oneqmc.molecule.Molecule or Sequence[oneqmc.molecule.Molecule]): the molecule
                or a sequence of molecules to compute Hartree-Fock parameters for.
            max_up (int): maximum number of up-spin electrons. Used to pad the tensors to the
                same shape
            max_down (int): maximum number of down-spin electrons. Used to pad the tensors to the
                same shape
            basis (str): the name of a Gaussian basis set, recognised by pyscf
            max_n_gaussians (int): the maximum number of primitive Gaussians per atomic orbital
                function. Should match with the basis name.
            pyscf_kwargs (dict): additional keyword arguments to pyscf
        Returns:
            idxs (jax.Array of shape (num_mols, max_orbitals)): indicates which atom is used as
                the centre of each orbital
            shells (GaussianAtomicOrbitalSpecification batched over (num_mols, max_orbitals)):
                the specification of each orbital shell
            mo_coeff_up (jax.Array of shape (num_mols, max_orbitals, max_up)): coefficients
                to contract over the atomic orbitals to produce molecular orbitals (up).
            mo_coeff_down (jax.Array of shape (num_mols, max_orbitals, max_down)): coefficients
                to contract over the atomic orbitals to produce molecular orbitals (down).
        """
        pyscf_inputs = [(mol.as_pyscf(), int(mol.charge), int(mol.spin), basis) for mol in mols]
        pyscf_outputs = Parallel(n_jobs=-1)(
            delayed(pyscf_from_mol)(*pyscf_input, **pyscf_kwargs) for pyscf_input in pyscf_inputs
        )

        scf_params = []
        for mol, output in tqdm(
            zip(mols, pyscf_outputs), total=len(mols), desc="Processing SCF output"
        ):
            if output is None:
                raise ValueError(f"Failed to compute SCF for molecule {mol}")
            mol_py, coeff, overlap = output
            idxs, shells = gto_spec_from_pyscf(mol_py, max_n_gaussians=max_n_gaussians)
            mo_coeff = jnp.asarray(coeff)
            ao_overlap = jnp.asarray(overlap)
            mo_coeff *= jnp.sqrt(jnp.diag(ao_overlap))[:, None]
            conf_up, conf_down = [jnp.arange(n_el) for n_el in (mol.n_up, mol.n_down)]
            mo_coeff_up = zero_embed(mo_coeff[..., conf_up], dims.max_up)
            mo_coeff_down = zero_embed(mo_coeff[..., conf_down], dims.max_down)
            scf_params.append((idxs, shells, mo_coeff_up, mo_coeff_down))

        max_orbitals = max(idxs.shape[-1] for (idxs, *_) in scf_params)
        embed_scf_params = []
        for (idxs, shells, mo_coeff_up, mo_coeff_down) in scf_params:
            mo_coeff_up = zero_embed(mo_coeff_up, max_orbitals, axis=-2)
            mo_coeff_down = zero_embed(mo_coeff_down, max_orbitals, axis=-2)
            idxs = zero_embed(idxs, max_orbitals)
            shells = shells.embed_to_size(max_orbitals)
            embed_scf_params.append((idxs, shells, mo_coeff_up, mo_coeff_down))
        return embed_scf_params
