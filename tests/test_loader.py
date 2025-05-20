from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import pytest

from oneqmc.convert_geo import load_molecules
from oneqmc.data import (
    Batch,
    as_dict_stream,
    as_mol_conf_stream,
    chunkify,
    key_chain,
    merge_dicts,
    simple_batch_loader,
)
from oneqmc.molecule import Molecule
from oneqmc.preprocess.augmentation import Augmentation
from oneqmc.types import ModelDimensions, ScfParams
from oneqmc.wf.transferable_hf.hf import HartreeFock


@pytest.fixture
def molecules() -> Sequence[Molecule]:
    path = f"{Path(__file__).resolve().parent.parent}/data/cyclobutadiene"
    return tuple(load_molecules(path))


@pytest.fixture
def dims(molecules: Sequence[Molecule]) -> ModelDimensions:
    return ModelDimensions.from_molecules(molecules)


@pytest.fixture
def scf_params(molecules: Sequence[Molecule], dims: ModelDimensions) -> ScfParams:
    return HartreeFock.from_mol(molecules, dims)


def create_loader(
    molecules: Sequence[Molecule],
    dims: ModelDimensions,
    scf_params: ScfParams | None = None,
    batch_size: int = 1,
    shuffle: bool = False,
    augmentations: Sequence[Augmentation] = (),
    repeat: bool = False,
    with_scf: bool = False,
    force_sequence: bool = False,
) -> Iterable[Batch]:
    rng = jax.random.PRNGKey(0)

    streams: list[Iterable[dict[str, Any]]] = [
        as_dict_stream("mol", as_mol_conf_stream(dims, molecules))
    ]
    if with_scf:
        if scf_params is None:
            raise ValueError("scf_params must be provided when with_scf is True")
        streams.append(as_dict_stream("scf", scf_params))

    merged_streams = map(merge_dicts, zip(*streams, strict=True))
    if force_sequence:
        merged_streams = tuple(merged_streams)  # type: ignore

    rng, key = jax.random.split(rng)
    loader = simple_batch_loader(
        merged_streams,  # type: ignore
        batch_size,
        rng=key if shuffle else None,
        repeat=repeat,
    )

    for augmentation in augmentations:
        rng, key = jax.random.split(rng)
        loader = map(  # type: ignore
            lambda rng_and_batch: augmentation(*rng_and_batch),
            zip(key_chain(key), loader),
        )

    return loader


def test_cannot_shuffle_stream(
    molecules: Sequence[Molecule],
    dims: ModelDimensions,
):
    with pytest.raises(ValueError, match="only supported for sequence"):
        loader = create_loader(molecules, dims, shuffle=True)
        next(iter(loader))


def test_shuffle(
    molecules: Sequence[Molecule],
    dims: ModelDimensions,
):
    loader = create_loader(molecules, dims, shuffle=True, force_sequence=True)
    idxs = []
    for idx, _ in loader:
        assert isinstance(idx, jax.Array)
        assert idx.shape == (1, 1)
        idxs.append(int(idx.flatten()[0]))
    assert len(set(idxs)) == len(idxs)


def test_chunkify():
    data = [1, 2, 3, 4, 5, 6, 7]
    chunks = [[1, 2, 3], [4, 5, 6], [7]]
    for chunk, ref in zip(chunkify(data, 3, strict=False), chunks, strict=True):
        assert tuple(chunk) == tuple(ref)

    with pytest.raises(ValueError, match="must be positive"):
        tuple(chunkify(data, 0))

    with pytest.raises(ValueError, match="less than"):
        for chunk, ref in zip(chunkify(data, 2, strict=True), chunks, strict=True):
            pass


def unbatch(x):
    return jax.tree_util.tree_map(lambda y: y[0, 0], x)


def test_correct_mol_confs(
    molecules: Sequence[Molecule],
    dims: ModelDimensions,
):
    reference = [mol.to_mol_conf(dims.max_nuc) for mol in molecules]
    loader = create_loader(molecules, dims)
    collected = [unbatch(batch["mol"]) for _, batch in loader]
    for ref, col in zip(reference, collected, strict=True):
        assert all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.allclose, ref, col)))


def test_correct_scfs(
    molecules: Sequence[Molecule],
    dims: ModelDimensions,
    scf_params: ScfParams,
):
    loader = create_loader(molecules, dims, scf_params, with_scf=True)
    collected = [unbatch(batch["scf"]) for _, batch in loader]
    for ref, col in zip(scf_params, collected, strict=True):
        assert all(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.allclose, ref, col)))


def test_repeat(
    molecules: Sequence[Molecule],
    dims: ModelDimensions,
):
    loader = create_loader(molecules, dims, repeat=True)
    num_elems = len(molecules)
    num_taken = 2 * num_elems
    collected = []
    for _, batch in zip(range(num_taken), loader):
        collected.append(batch)
    assert len(collected) == num_taken
    for i in range(num_elems):
        assert collected[i][0] == collected[i + num_elems][0]
        assert jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(jnp.allclose, collected[i], collected[i + num_elems])
        )
    loader = create_loader(molecules, dims, repeat=False)
    loader_iter = iter(loader)
    with pytest.raises(StopIteration):
        for _ in range(num_taken):
            next(loader_iter)


def test_rewind(
    molecules: Sequence[Molecule],
    dims: ModelDimensions,
):
    loader = create_loader(molecules, dims, repeat=False)
    num_elems = len(molecules)
    collected = []
    for _, batch in zip(range(num_elems), loader):
        collected.append(batch)


@pytest.mark.parametrize("batch_size", [1, 3, 5, 6, 12])
def test_batch_size(molecules: Sequence[Molecule], dims: ModelDimensions, batch_size: int):
    num_devices = 1
    batch_shape = (num_devices, batch_size)
    if batch_size > len(molecules) or len(molecules) % batch_size != 0:
        with pytest.raises(ValueError, match="less than"):
            loader = create_loader(molecules, dims, batch_size=batch_size)
            tuple(loader)

        loader = create_loader(molecules, dims, batch_size=batch_size, repeat=True)
        for _, (idx, data) in zip(range(2 * len(molecules)), loader):
            assert isinstance(idx, jax.Array)
            assert idx.shape == batch_shape
            assert all(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(lambda x: x.shape[:2] == batch_shape, data)
                )
            )
    else:
        loader = create_loader(molecules, dims, batch_size=batch_size)
        for _, (idx, data) in zip(range(len(molecules)), loader):
            assert isinstance(idx, jax.Array)
            assert idx.shape == batch_shape
            assert all(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(lambda x: x.shape[:2] == batch_shape, data)
                )
            )
