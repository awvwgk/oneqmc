import itertools as it
from typing import Any, Callable, Generator, Iterable, Sequence, TypeAlias, TypeVar

import jax
import jax.numpy as jnp

from .molecule import Molecule
from .types import ModelDimensions, MolecularConfiguration, RandomKey

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")

IntegerLike: TypeAlias = int | jax.Array
BatchData: TypeAlias = dict[str, Any]
Batch: TypeAlias = tuple[IntegerLike, BatchData]
DataLoader: TypeAlias = Iterable[Batch]


def scan(
    f: Callable[[T, S], tuple[T, U]],
    init: T,
    stream: Iterable[S],
) -> Generator[U, None, None]:
    """Scan a function over a stream of elements. Same semantics as jax.lax.scan."""
    state = init
    for elem in stream:
        state, out = f(state, elem)
        yield out


def chunkify(
    iterable: Iterable[T], n: int, *, strict: bool = True
) -> Generator[Sequence[T], None, None]:
    """Split an iterable into chunks of size n."""
    if n < 1:
        raise ValueError(f"Chunk size {n} must be positive")
    iterator = iter(iterable)
    while True:
        chunk = tuple(it.islice(iterator, n))
        if len(chunk) == 0:
            break
        elif len(chunk) != n and strict:
            raise ValueError(f"Number of elements {len(chunk)} is less than chunk size {n}")
        yield chunk


class ShuffledStream(Iterable[T]):
    """Shuffled stream of elements from a source iterable.

    Parameters
    ----------
    source : Iterable[T] | Sequence[T]
        The source iterable to shuffle.
    rng : RandomKey
        The random key to use for shuffling.
    buffer_size : int | None
        The number of elements to buffer before shuffling. If None, the source must be a sequence.
    """

    def __init__(
        self,
        source: Iterable[T] | Sequence[T],
        rng: RandomKey,
        buffer_size: int | None = None,
    ):
        self.source = source
        self.rng = rng
        if isinstance(source, Sequence):
            self.source = iter(source)
            self.buffer_size = len(source)
        else:
            if buffer_size is None:
                raise ValueError("buffer_size must be specified for non-sequence sources")
            self.buffer_size = buffer_size

    def __iter__(self) -> Generator[T, None, None]:
        rng = self.rng
        indices = jnp.arange(self.buffer_size)
        for buffer in chunkify(self.source, self.buffer_size):
            rng, key = jax.random.split(rng)
            indices = jax.random.permutation(key, indices, independent=True)
            for idx in indices:
                yield buffer[idx]


class PyTreeBatcher(Iterable[T]):
    """Batcher for pytrees.

    Batches are split along the first axis. The first axis must be divisible by the number of devices.

    Parameters
    ----------
    source : Iterable[T]
        The source iterable.
    batch_size : int
        The batch size.
    """

    def __init__(
        self,
        source: Iterable[T],
        batch_size: int,
    ):
        if batch_size % jax.local_device_count() != 0:
            raise ValueError(
                f"Batch size {batch_size} is not divisible by the number of devices {jax.local_device_count()}"
            )
        if batch_size < jax.local_device_count():
            raise ValueError(
                f"Batch size {batch_size} is less than the number of devices {jax.local_device_count()}"
            )
        self.source = source
        self.batch_size = batch_size

    def __iter__(self) -> Generator[T, None, None]:
        @jax.jit
        def pack(*xs):
            return jax.tree_util.tree_map(
                lambda *xs: jnp.reshape(
                    jnp.stack(xs, axis=0),
                    (jax.local_device_count(), -1, *jnp.shape(xs[0])),
                ),
                *xs,
            )

        for elems in chunkify(self.source, self.batch_size, strict=True):
            num_taken = len(elems)
            if num_taken < self.batch_size:
                raise ValueError(
                    f"Number of elements {num_taken} is less than the batch size {self.batch_size}"
                )
            batch: T = pack(*elems)
            yield batch


Key = TypeVar("Key")


def merge_dicts(dicts: Iterable[dict[Key, Any]]) -> dict[Key, Any]:
    """Merge a stream of dictionaries into a single dictionary."""
    return dict(it.chain.from_iterable(d.items() for d in dicts))


def as_mol_conf_stream(
    dims: ModelDimensions, mols: Iterable[Molecule]
) -> Generator[MolecularConfiguration, None, None]:
    """Convert a stream of molecules into a stream of molecular configurations."""
    for mol in mols:
        yield mol.to_mol_conf(dims.max_nuc)


def as_dict_stream(key: str, stream: Iterable[T]) -> Generator[dict[str, T], None, None]:
    """Convert a stream of elements into a stream of dictionaries with a single key."""
    for elem in stream:
        yield {key: elem}


def key_chain(key: RandomKey) -> Generator[RandomKey, None, None]:
    """Generate a stream of random keys."""
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def simple_batch_loader(
    source: Iterable[BatchData] | Sequence[BatchData],
    batch_size: int,
    rng: RandomKey | None = None,
    repeat: bool = True,
) -> Generator[Batch, None, None]:
    """Batch loader for simple data sources.

    Parameters
    ----------
    source : Iterable[BatchData] | Sequence[BatchData]
        The source iterable.
    batch_size : int
        The batch size.
    rng : RandomKey | None
        The random key to use for shuffling. If None, no shuffling is performed.
        If the source is a sequence, the buffer size is set to the length of the source.
        Otherwise raise an error.
    repeat : bool
        Whether to repeat the source indefinitely.
    """
    stream: Iterable[Batch] = enumerate(source)
    if repeat:
        stream = it.cycle(stream)
    if rng is not None:
        if isinstance(source, Sequence):
            shuffle_buffer_size = len(source)
            stream = ShuffledStream(stream, rng, buffer_size=shuffle_buffer_size)
        else:
            raise ValueError("Shuffling is only supported for sequence sources")
    yield from PyTreeBatcher(stream, batch_size)
