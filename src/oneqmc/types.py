from collections import namedtuple
from typing import Any, Dict, Generic, Mapping, Sequence, Tuple, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jax.nn import softmax
from typing_extensions import Self

Psi = namedtuple("Psi", "sign log")
Stats: TypeAlias = Mapping[str, Any]
Energy: TypeAlias = jax.Array
EnergyAndGradMask: TypeAlias = Tuple[Energy, jax.Array]
Weight: TypeAlias = jax.Array
Mask: TypeAlias = jax.Array
WavefunctionParams: TypeAlias = Dict
InitialiserParams: TypeAlias = Dict
ScfParams: TypeAlias = Sequence[tuple[Any, Any, Any, Any]]
RandomKey: TypeAlias = jax.Array
OptimizerState: TypeAlias = Dict
TrainState = namedtuple("TrainState", "sampler params opt")

ArrayType = TypeVar("ArrayType", jax.Array, np.ndarray)


@jdc.pytree_dataclass
class ModelDimensions:
    """Model dimensions dataclass.

    Attributes
    ----------
    max_nuc: int
        The maximum number of nuclei.
    max_up: int
        The maximum number of spin-up electrons.
    max_down: int
        The maximum number of spin-down electrons.
    max_charge: int
        The maximum atom charge of the dataset.
    max_species: int
        The maximum atomic species of the dataset.
    """

    max_nuc: int
    max_up: int
    max_down: int
    max_charge: int
    max_species: int

    @staticmethod
    def from_molecules(
        molecules,  #: Sequence[Molecule],
        increment_max_nuc: int = 0,
        increment_max_up: int = 0,
        increment_max_down: int = 0,
        increment_max_charge: int = 0,
        increment_max_species: int = 0,
    ) -> "ModelDimensions":
        max_nuc, max_up, max_down, max_charge, max_species = map(
            max,
            zip(
                *[
                    (mol.n_nuc, mol.n_up, mol.n_down, max(mol.charges), max(mol.species))
                    for mol in molecules
                ]
            ),
        )
        max_nuc += increment_max_nuc  # type: ignore
        max_up += increment_max_up  # type: ignore
        max_down += increment_max_down  # type: ignore
        max_charge += increment_max_charge  # type: ignore
        max_species += increment_max_species  # type: ignore
        return ModelDimensions(max_nuc, max_up, max_down, max_charge, max_species)  # type: ignore

    def to_dict(self):
        return {k: int(v) for k, v in vars(self).items()}


@jdc.pytree_dataclass
class Embeddings(Generic[ArrayType]):
    """Embeddings dataclass.

    Attributes
    ----------
    vector: jax.Array of shape (max_nuc, 3, f)
    charges: jax.Array of shape (max_nuc, f)
    """

    vector: ArrayType
    scalar: ArrayType

    # mypy doesn't like generic dataclasses
    @staticmethod
    def make(vector: ArrayType, scalar: ArrayType):
        return Embeddings(vector, scalar)  # type: ignore


@jdc.pytree_dataclass
class Nuclei:
    """Nuclei dataclass.

    Attributes
    ----------
    coords : jax.Array of shape (max_nuc, 3)
        Nuclear coordinates as rows.
    charges : jax.Array of shape (max_nuc, )
        Atom charges.
    n_active : jax.Array of shape ()
        How many nuclei are active (i.e. not masked).
    """

    coords: jax.Array  # Shape (max_nuc, 3)
    charges: jax.Array  # Shape (max_nuc, )
    species: jax.Array  # Shape (max_nuc, )
    n_active: jax.Array  # Shape ()
    embeddings: Embeddings[jax.Array] | None = None

    @staticmethod
    def embed_to_masked(
        coords: jax.Array,  # Shape (n_nuc, 3)
        charges: jax.Array,  # Shape (n_nuc, )
        species: jax.Array,  # Shape (n_nuc, )
        embeddings: Embeddings[jax.Array] | None = None,
        size: int | None = None,
    ):
        """Create a Nuclei object from coordinates and charges and optionally mask it to a fixed size.

        Parameters
        ----------
        coords : jax.Array of shape (n_nuc, 3)
            Nuclear coordinates as rows.
        charges : jax.Array of shape (n_nuc, )
            Atom charges.
        size : int | None, optional
            The embedding dimension to which the nuclei should be masked, by default None.
            If None or equal to the number of nuclei, all nuclei are considered active.
            Raises a ValueError if size is smaller than the number of nuclei.

        Returns
        -------
        Nuclei
            Nuclei object
        """
        with jax.ensure_compile_time_eval():
            n_active = jnp.array(len(coords))
        if size is not None:
            if size > len(coords):
                coords = jnp.zeros((size, 3), dtype=coords.dtype).at[: len(coords)].set(coords)
                charges = jnp.zeros((size,), dtype=charges.dtype).at[: len(charges)].set(charges)
                species = jnp.zeros((size,), dtype=charges.dtype).at[: len(species)].set(species)
                if embeddings is not None:
                    embeddings = jdc.replace(
                        embeddings,
                        vector=jnp.zeros(
                            (size, embeddings.vector.shape[1], embeddings.vector.shape[2]),
                            dtype=embeddings.vector.dtype,
                        )
                        .at[:n_active]
                        .set(embeddings.vector),
                        scalar=jnp.zeros(
                            (size, embeddings.scalar.shape[1]), dtype=embeddings.scalar.dtype
                        )
                        .at[:n_active]
                        .set(embeddings.scalar),
                    )
            elif size < len(coords):
                raise ValueError(
                    f"Embedding dimension {size} is smaller than the number of nuclei {len(coords)}"
                )
        return Nuclei(coords, charges, species, n_active, embeddings)  # type: ignore

    @property
    def count(self):
        return self.n_active

    @property
    def mask(self) -> jax.Array:
        """Get the mask. If nuclei are not masked, raise an error."""
        return jnp.arange(self.max_count) < jnp.expand_dims(self.n_active, -1)

    @property
    def max_count(self):
        return self.coords.shape[-2]

    @property
    def batch_shape(self):
        return self.coords.shape[:-2]

    @property
    def total_charge(self):
        """Get the total charge of the nuclei."""
        return jnp.sum(self.mask * self.charges, axis=-1)


@jdc.pytree_dataclass
class MolecularConfiguration:
    r"""Represent input physical configurations of nuclei.

    Parameters
    ----------
    nuclei : Nuclei
        Nuclei coordinates and charges.
    total_charge : int
        Total charge of the system.
    total_spin : int
        Total spin of the system.


    Attributes
    ----------
    n_nuc : int
        Number of nuclei.
    n_up : int
        Number of spin-up electrons.
    n_down : int
        Number of spin-down electrons.
    """

    nuclei: Nuclei
    total_charge: jax.Array
    total_spin: jax.Array

    @property
    def max_nuc(self):
        return self.nuclei.max_count

    @property
    def n_nuc(self):
        return self.nuclei.count

    @property
    def n_up(self):
        return (self.nuclei.total_charge - self.total_charge + self.total_spin) // 2

    @property
    def n_down(self):
        return (self.nuclei.total_charge - self.total_charge - self.total_spin) // 2


@jdc.pytree_dataclass
class ParallelElectrons:
    """Electrons dataclass, representing electrons of parallel spin.

    Attributes
    ----------
    coords : jax.Array of shape (max_elec, 3)
        Nuclear coordinates as rows.
    n_active : jax.Array of shape ()
        How many electrons are active (i.e. not masked).
    """

    coords: jax.Array  # Shape (max_elec, 3)
    n_active: jax.Array  # Shape ()

    @property
    def count(self) -> jax.Array:
        return self.n_active

    @property
    def max_count(self):
        return self.coords.shape[-2]

    @property
    def mask(self) -> jax.Array:
        """Get the mask. If electrons are not masked."""
        return jnp.arange(self.max_count) < jnp.expand_dims(self.n_active, -1)


@jdc.pytree_dataclass
class ElectronConfiguration:
    r"""Represents configurations of electrons.

    Parameters
    ----------
    up : Electrons
        Spin-up electrons.
    down : Electrons
        Spin-down electrons.

    Attributes
    ----------
    count : jax.Array
        Number of electrons present (unmasked).
    max_elec: int
        Number of electrons representable (masked + unmasked).
    coords: jax.Array
        Spatial coordinates of electrons.
    mask: jax.Array
        Mask indicating existence of electrons.
    n_up : jax.Array
        Number of spin-up electrons (unmasked).
    n_down : jax.Array
        Number of spin-down electrons (unmasked).
    max_up: int
        Number of up-spin electrons representable (masked + unmasked).
    max_down: int
        Number of down-spin electrons representable (masked + unmasked).
    spin: jax.Array
        Array of spins of electrons coded as up=+1, down=-1.
    """

    up: ParallelElectrons
    down: ParallelElectrons

    @property
    def count(self):
        return self.up.count + self.down.count

    @property
    def max_elec(self):
        return self.up.max_count + self.down.max_count

    @property
    def coords(self):
        return jnp.concatenate((self.up.coords, self.down.coords), axis=-2)

    @property
    def mask(self):
        return jnp.concatenate((self.up.mask, self.down.mask), axis=-1)

    @property
    def n_up(self):
        return self.up.count

    @property
    def n_down(self):
        return self.down.count

    @property
    def max_up(self):
        return self.up.max_count

    @property
    def max_down(self):
        return self.down.max_count

    @property
    def spins(self):
        return jnp.concatenate((jnp.ones(self.max_up), -jnp.ones(self.max_down)), axis=-1)

    def update(self, coords: jax.Array) -> Self:
        return jdc.replace(
            self,
            up=jdc.replace(self.up, coords=coords[..., : self.max_up, :]),
            down=jdc.replace(self.down, coords=coords[..., self.max_up :, :]),
        )


@jdc.pytree_dataclass
class WeightedElectronConfiguration:
    r"""Represents configurations of electrons with statistical weights from sampling.

    Parameters
    ----------
    elec_conf : ElectronConfiguration
        Spin-up electrons.
    log_weight : jax.Array
        *Unnormalised* log weights
    """

    elec_conf: ElectronConfiguration
    log_weight: jax.Array

    @classmethod
    def uniform_weight(cls, elec_conf: ElectronConfiguration):
        log_weight = jnp.zeros(elec_conf.count.shape)
        return cls(elec_conf, log_weight)  # type: ignore

    @property
    def coords(self):
        return self.elec_conf.coords

    def n_normed_weight(self, axis=0):
        """Return weights that are normalised to sum to the length of the dimension.

        For example, uniform weights are normalised to all ones.

        Parameters
        ----------
        axis : int
            The axis to apply normalisation over
        """
        return softmax(self.log_weight, axis=axis) * self.log_weight.shape[axis]

    @property
    def mask(self):
        return self.elec_conf.mask
