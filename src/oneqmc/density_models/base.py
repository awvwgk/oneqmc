from typing import Any, Dict, Generic, NamedTuple, Protocol, Tuple, TypeVar

import jax

from ..types import MolecularConfiguration, RandomKey, Stats, WavefunctionParams

B = TypeVar("B", covariant=True)
_B = TypeVar("_B", contravariant=True)
O = TypeVar("O")
P = TypeVar("P")


class DensityModel(Generic[P], Protocol):
    def init(self, rng: RandomKey, x: jax.Array, mol: MolecularConfiguration) -> P:
        ...

    def apply(self, params: P, x: jax.Array, mol: MolecularConfiguration) -> jax.Array:
        ...


class DensityMatrixModel(Generic[P], Protocol):
    def init(self, rng: RandomKey, x: jax.Array, xp: jax.Array) -> P:
        ...

    def apply(self, params: P, x: jax.Array, xp: jax.Array) -> jax.Array:
        ...


class DensityFittingBatchFactory(Generic[B], Protocol):
    def __call__(
        self,
        rng: RandomKey,
        smpl_state: dict,
        params: WavefunctionParams,
        inputs: Dict,
    ) -> B:
        ...

    def initial_sample(self) -> B:
        ...


class OneElectronSampler(Protocol):
    def sample_and_log_prob(self, *, seed: RandomKey, sample_shape: Tuple) -> Tuple:
        ...


class DensityMatrixTrainer(Generic[P, O, _B], Protocol):
    model: DensityMatrixModel[P]

    def __init__(self, model: DensityMatrixModel[P], **kwargs: Any):
        self.model = model
        ...

    def init(
        self, rng: RandomKey, mol: MolecularConfiguration, x: jax.Array, xp: jax.Array
    ) -> Tuple[P, O]:
        ...

    def step(self, rng: RandomKey, params: P, opt_state: O, batch: _B) -> Tuple[P, O, Stats]:
        ...


class DensityTrainer(Generic[P, O, _B], Protocol):
    model: DensityModel[P]

    def __init__(self, model: DensityModel[P], **kwargs: Any):
        self.model = model
        ...

    def init(self, rng: RandomKey, mol: MolecularConfiguration, x: jax.Array) -> Tuple[P, O]:
        ...

    def step(self, rng: RandomKey, params: P, opt_state: O, batch: _B) -> Tuple[P, O, Stats]:
        ...


class DensityFittingState(NamedTuple, Generic[P, O]):
    params: P
    opt_state: O
