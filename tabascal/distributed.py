"""Multi-process / multi-GPU support for ``tabascal run`` (Strategy C).

tabascal's MAP solve is a single global optimization. To scale it across several
GPUs -- one process per GPU under SLURM ``srun`` -- we shard every *per-baseline*
array along the baseline axis and replicate everything per-antenna. The likelihood
is independent per ``(baseline, freq, time)`` (see ``components/likelihood.py``), so
the only collective is a single all-reduce of the summed log-density that JAX/GSPMD
inserts automatically; the result is one *exact* MAP solution.

This module is the single home for the distributed plumbing so the rest of the code
stays legible. Nothing here has any effect when only one device is visible
(``jax.device_count() == 1``): ``sharding_enabled()`` is then ``False`` and callers
take the original single-device path.

Two execution modes are supported and share all the sharding logic:

* **Multi-process** (the target): launched by ``srun`` with one process per GPU.
  ``init_distributed()`` calls ``jax.distributed.initialize()`` (SLURM auto-detect).
  Each process reads and holds only *its* baseline block, then assembles a globally
  addressable ``jax.Array`` via :func:`shard_bl` -- this is what gives true memory
  scaling (no process materializes the full ``n_bl`` data).
* **Single-process, multiple local devices**: e.g. CPU tests with
  ``--xla_force_host_platform_device_count=2`` or a single node's GPUs. Each process
  holds the full array and :func:`shard_bl` partitions it across local devices. Used
  for the equivalence tests; exercises the identical mesh / sharding / partitioning
  code paths without a coordinator.

The baseline axis is padded up to a multiple of the global device count
(:class:`BaselinePartition`). Padded baselines are flagged out and zero-valued, so
they never enter the likelihood and are sliced away before any results are written.
"""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Mesh axis name for the baseline partition.
BL_AXIS = "bl"


# ---------------------------------------------------------------------------
# Initialization / capability checks
# ---------------------------------------------------------------------------

def _slurm_ntasks() -> int:
    """Number of SLURM tasks in this step, or 1 when not under SLURM."""
    for var in ("SLURM_NTASKS", "SLURM_STEP_NUM_TASKS", "SLURM_NPROCS"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


def init_distributed() -> None:
    """Bring up the JAX distributed runtime when launched multi-process.

    Calls :func:`jax.distributed.initialize` (which auto-detects the SLURM
    coordinator, process count and id) **only** when more than one SLURM task is
    present, or when ``TABASCAL_FORCE_DISTRIBUTED`` is set. Outside SLURM -- a plain
    ``python`` invocation or a single-process multi-device test -- this is a no-op,
    so we never block waiting for a coordinator that will not appear.

    Must be called before any JAX array is created (it initializes the device
    backend), i.e. first thing in :func:`tabascal.scripts._run_tabascal_impl.run`.
    """
    if os.environ.get("TABASCAL_FORCE_DISTRIBUTED") or _slurm_ntasks() > 1:
        jax.distributed.initialize()


def sharding_enabled() -> bool:
    """True when more than one global device is visible, so we should shard."""
    return jax.device_count() > 1


def is_process_0() -> bool:
    """True on the single process responsible for logging and writing results."""
    return jax.process_index() == 0


# ---------------------------------------------------------------------------
# Baseline partition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselinePartition:
    """Describes how ``n_bl`` baselines are split across the global devices.

    The baseline axis is padded to ``n_bl_padded = n_dev * block`` so it divides
    evenly across the device mesh (a hard requirement for an even ``NamedSharding``).
    Padded baselines carry zero data and ``True`` flags, so they contribute nothing
    to the likelihood and are dropped on gather/write.

    ``start``/``stop`` index into the *padded* space for **this process's** block;
    ``valid_stop`` is the first padded index past the real baselines in this block
    (``== stop`` for fully-real blocks, ``<= start`` for an all-padding block).
    """
    n_bl: int          # true baseline count
    n_dev: int         # global device count == mesh size
    block: int         # per-device block length (padded)
    start: int         # this process's block start, in padded space
    stop: int          # this process's block stop, in padded space

    @property
    def n_bl_padded(self) -> int:
        return self.n_dev * self.block

    @property
    def valid_stop(self) -> int:
        return max(self.start, min(self.stop, self.n_bl))

    @property
    def n_valid(self) -> int:
        """Number of real (non-padding) baselines in this process's block."""
        return self.valid_stop - self.start

    @property
    def local_rows(self) -> int:
        """Rows along axis 0 that this process feeds to :func:`shard_bl`.

        Multi-process: one padded block. Single-process: the whole padded axis
        (the process owns every device), so it supplies all ``n_bl_padded`` rows.
        """
        return self.block if jax.process_count() > 1 else self.n_bl_padded


def bl_partition(n_bl: int) -> BaselinePartition:
    """Balanced, padded baseline partition for the current process.

    With one local device per process (the SLURM layout), process ``i`` owns device
    ``i`` and baseline block ``i``. We assert one local device in the multi-process
    case so the process->block mapping is unambiguous; single-process runs may hold
    several local devices and partition the full array across them.
    """
    n_dev = jax.device_count()
    block = -(-n_bl // n_dev)  # ceil division -> padded block length

    if jax.process_count() > 1:
        if jax.local_device_count() != 1:
            raise RuntimeError(
                "Distributed tabascal expects one device per process (one GPU per "
                f"SLURM task); this process sees {jax.local_device_count()}."
            )
        idx = jax.process_index()
        start = idx * block
        stop = start + block
    else:
        # Single process owns the whole (padded) baseline axis.
        start, stop = 0, n_dev * block

    return BaselinePartition(n_bl=n_bl, n_dev=n_dev, block=block, start=start, stop=stop)


# ---------------------------------------------------------------------------
# Mesh + sharding helpers
# ---------------------------------------------------------------------------

def make_bl_mesh() -> Mesh:
    """1-D mesh over **all** global devices, named for the baseline axis."""
    return Mesh(np.asarray(jax.devices()), (BL_AXIS,))


def _bl_spec(ndim: int) -> P:
    """PartitionSpec that shards axis 0 (baseline) and replicates the rest."""
    return P(BL_AXIS, *([None] * (ndim - 1)))


def replicate(x, mesh: Mesh):
    """Place ``x`` fully replicated on every device of ``mesh``."""
    return jax.device_put(x, NamedSharding(mesh, P()))


def pad_bl(local: np.ndarray, part: BaselinePartition, pad_value=0) -> np.ndarray:
    """Pad a process-local baseline-axis array up to ``part.local_rows`` rows.

    ``local`` holds the real baselines this process read (a block in multi-process,
    the full ``n_bl`` in single-process). The tail rows added here are the padding
    baselines; callers flag them out so they never enter the likelihood.
    """
    pad_rows = part.local_rows - local.shape[0]
    if pad_rows < 0:
        raise ValueError(
            f"local block has {local.shape[0]} baselines, exceeds expected "
            f"{part.local_rows}"
        )
    if pad_rows == 0:
        return local
    pad = [(0, pad_rows)] + [(0, 0)] * (local.ndim - 1)
    return np.pad(local, pad, constant_values=pad_value)


def shard_bl(local_block: np.ndarray, part: BaselinePartition, mesh: Mesh):
    """Assemble a baseline-sharded global ``jax.Array`` from process-local data.

    ``local_block`` is this process's already-padded block (length ``part.block``
    along axis 0). In the multi-process case each process contributes its own block
    via :func:`jax.make_array_from_process_local_data`; in the single-process case
    the caller passes the full padded array and we partition it across local devices.
    """
    global_shape = (part.n_bl_padded,) + tuple(local_block.shape[1:])
    sharding = NamedSharding(mesh, _bl_spec(len(global_shape)))

    if jax.process_count() > 1:
        return jax.make_array_from_process_local_data(sharding, local_block, global_shape)
    return jax.device_put(np.asarray(local_block), sharding)


def _leaf_sharding(x, n_bl: int, mesh: Mesh) -> NamedSharding:
    """Shard a leaf along baseline axis 0 iff its leading dim is ``n_bl``.

    Per-baseline arrays (``vis_*`` state, ``ast_k_*`` params, ``a1``/``a2``,
    ``sigma_ast_k``/``mu_ast_k``) lead with ``n_bl`` (the padded baseline count) and
    are sharded; everything else -- per-antenna params/constants, GP kernels, scalars
    -- has a different (or no) leading dim and is replicated. ``n_bl`` is the padded
    count, which for an interferometer (``~n_ant*(n_ant-1)/2``) never coincides with
    ``n_ant`` or ``n_rfi``, so the shape rule is unambiguous.
    """
    ndim = getattr(x, "ndim", 0)
    spec = _bl_spec(ndim) if (ndim >= 1 and x.shape[0] == n_bl) else P()
    return NamedSharding(mesh, spec)


def put_leaf(x, n_bl: int, mesh: Mesh):
    """Place one leaf on the mesh, sharded-or-replicated per :func:`_leaf_sharding`."""
    return jax.device_put(x, _leaf_sharding(x, n_bl, mesh))


def shard_pytree(tree, n_bl: int, mesh: Mesh):
    """Apply :func:`put_leaf` to every array leaf of ``tree``."""
    return jax.tree_util.tree_map(lambda x: put_leaf(x, n_bl, mesh), tree)


def gather_bl(arr) -> np.ndarray:
    """Gather a baseline-sharded array to a full host array on every process.

    Returns the concatenated ``(n_bl_padded, ...)`` data; callers slice ``[:n_bl]``
    to drop padding. Used on process 0 before writing zarr/MS results.
    """
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(arr, tiled=True))


def to_host(arr) -> np.ndarray:
    """Materialize a (possibly sharded) device array as a full host numpy array.

    Multi-process: gather every process's shard via ``process_allgather``.
    Single-process (one controller, any number of local devices): ``np.asarray``
    already assembles the local shards. Used when writing results / computing
    host-side metrics, so callers see the whole array regardless of sharding.
    """
    if jax.process_count() > 1:
        return gather_bl(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Process-0 IO guards
# ---------------------------------------------------------------------------

def print0(*args, **kwargs) -> None:
    """``print`` only on process 0; a no-op elsewhere."""
    if is_process_0():
        print(*args, **kwargs)


@contextmanager
def suppress_worker_stdout():
    """Silence stdout on non-zero processes for the duration of the block.

    Component ``setup`` and the model summary print a lot; without this every worker
    would duplicate it. Process 0 is untouched.
    """
    if is_process_0():
        yield
        return
    saved = sys.stdout
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = saved
