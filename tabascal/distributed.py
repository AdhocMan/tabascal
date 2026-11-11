import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:  # public since jax 0.8; the experimental path covers the 0.6/0.7 range of our pin
    from jax import shard_map
except ImportError:  # pragma: no cover - exercised only on older jax
    from jax.experimental.shard_map import shard_map


# ---------------------------------------------------------------------------
# Initialization / capability checks
# ---------------------------------------------------------------------------

def _world_size() -> int:
    """Number of processes in this launch, across known launchers; 1 if single-process.

    Reads the world-size variable exported by whichever launcher started us -- SLURM,
    OpenMPI, MPICH/Intel MPI (PMI) or ``torchrun``/torch-elastic. We cannot infer
    multi-process from ``CUDA_VISIBLE_DEVICES`` alone: it tells us a GPU was pinned, not
    that peers exist or how to reach the coordinator. Returns 1 when no such variable is
    set, i.e. a plain ``python`` invocation.
    """
    for var in ("SLURM_NTASKS", "SLURM_STEP_NUM_TASKS", "SLURM_NPROCS",
                "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


def _process_rank() -> int:
    """This process's rank, across known launchers; 0 if unset."""
    for var in ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 0


def init_distributed() -> None:
    """Bring up the JAX distributed runtime when launched multi-process.

    Calls :func:`jax.distributed.initialize` **only** when the launcher reports more
    than one process (:func:`_world_size`), or when ``TABASCAL_FORCE_DISTRIBUTED`` is
    set. Outside a multi-process launch -- a plain ``python`` invocation or a
    single-process multi-device test -- this is a no-op, so we never block waiting for a
    coordinator that will not appear.

    Under SLURM (or when no MPI-style coordinator address is exported) we let JAX
    auto-detect the coordinator, process count and id. For other launchers that export
    ``MASTER_ADDR``/``MASTER_PORT`` (``torchrun`` and friends) we pass explicit
    coordinates. After this call ``jax.device_count()`` reports the *global* device
    count, so :func:`sharding_enabled` and everything downstream turn on automatically.

    Must be called before any JAX array is created (it initializes the device backend);
    the CLI calls it in ``run_tabascal._run_cmd`` before even importing the run
    implementation module.
    """
    if not (os.environ.get("TABASCAL_FORCE_DISTRIBUTED") or _world_size() > 1):
        return

    # SLURM auto-detects; so does JAX when no MPI-style coordinator is exported.
    if os.environ.get("SLURM_NTASKS") or "MASTER_ADDR" not in os.environ:
        jax.distributed.initialize()
    else:
        jax.distributed.initialize(
            coordinator_address=f"{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '1234')}",
            num_processes=_world_size(),
            process_id=_process_rank(),
        )


def sharding_enabled() -> bool:
    """True when more than one global device is visible, so we should shard.

    ``jax.device_count()`` is the *global* device count once :func:`init_distributed`
    has run (it is the *local* count before that). So in the one-GPU-per-process layout
    this returns ``True`` on every process after init, even though each process owns a
    single device.
    """
    return jax.device_count() > 1


def is_process_0() -> bool:
    """True on the single process responsible for logging and writing results."""
    return jax.process_index() == 0


def process_count() -> int:
    """Number of processes in this run; 1 outside a multi-process launch."""
    return jax.process_count()

# ---------------------------------------------------------------------------
# Baseline-axis sharding
# ---------------------------------------------------------------------------
#
# A sharded run splits the visibility axis. Each device owns one contiguous block
# of baselines, in the order the data already has, computes only those rows and
# holds only those rows of every visibility-shaped array; nothing visibility-sized
# is ever summed across devices, only the scalar likelihood is. Per-source and
# per-antenna arrays (the RFI signal and phase, the gains) carry no baseline axis
# and are replicated: every device needs all of them to form its own baselines.

#: The one axis of the device mesh.
BASELINE_AXIS = "bl"

# Array names whose *leading* axis is the baseline axis and which are therefore
# split across devices. Everything else (per-source RFI arrays, per-antenna gains,
# component constants) is replicated. Constants are keyed "<prefix>/<name>" in the
# model, so matching happens on the part after the last "/". Membership is not
# sufficient on its own: the leading dimension must also be the baseline count, so
# an array whose first axis merely shares the name is never split by accident, and
# that count must divide by the device count (see :func:`baselines_divide`).
BASELINE_AXIS_NAMES = frozenset({
    # the visibilities the components accumulate into, and their sum
    "vis_ast", "vis_rfi", "vis_obs",
    # ast_vis:GPVisAst's per-baseline latents and their prior
    "ast_k_r_base", "ast_k_i_base", "sigma_ast_k", "mu_ast_k",
})

#: Prefix of a constant that holds one row per *device* rather than per baseline:
#: the index tables each device hands the RFI visibility kernels for its own
#: baselines (see :func:`split_baselines`). Its leading axis is the device count,
#: split one row to a device.
DEVICE_ROWS_PREFIX = "dev_"


@lru_cache(maxsize=None)
def mesh() -> Mesh:
    """The single 1-D device mesh, one axis named ``"bl"``.

    Spans *all* global devices, which covers both the single-process multi-GPU
    layout and the multi-process one-GPU-per-process layout uniformly. Cached:
    every caller must use the same mesh object so shardings compare equal and
    jit avoids spurious recompiles.
    """
    return Mesh(np.array(jax.devices()), (BASELINE_AXIS,))


def baseline_sharding() -> NamedSharding:
    """Split an array's leading axis -- baselines, or device rows -- over the mesh."""
    return NamedSharding(mesh(), P(BASELINE_AXIS))


def replicated_sharding() -> NamedSharding:
    """Sharding that replicates an array on every device of the mesh."""
    return NamedSharding(mesh(), P())


def baselines_divide(n_bl: int) -> bool:
    """Whether a sharded run can split ``n_bl`` baselines evenly over the devices.

    It holds for a full array of an even device count -- ``n(n-1)/2`` for 256,
    384 and 512 stations is 32640, 73536 and 130816, each a multiple of four --
    but not necessarily for a flagged or selected subset. Where it fails, the
    visibility-shaped arrays stay replicated between components: the run is
    still correct, and still divides the RFI visibility work, but saves no
    memory on them. False whenever sharding is off.
    """
    return sharding_enabled() and n_bl % jax.device_count() == 0


def make_global(x, sharding: NamedSharding) -> jax.Array:
    """Globalize a full per-process copy, or reshard an existing global array.

    Every process holds the complete array (they all read the same MS / run the
    same setup), so the callback just slices the requested shard out of the
    local copy. This is the supported construction path in both single-process
    multi-device and multi-process layouts, unlike ``jax.device_put`` which
    historically rejects cross-process shardings.

    A global array that is already on the mesh is resharded collectively rather
    than fetched through numpy, which a process holding only part of it cannot do.
    """
    if isinstance(x, jax.Array):
        if x.sharding == sharding:
            return x
        if getattr(x.sharding, "mesh", None) == sharding.mesh:
            return jax.jit(lambda value: value, out_shardings=sharding)(x)
    x_np = np.asarray(x)
    return jax.make_array_from_callback(x_np.shape, sharding, lambda idx: x_np[idx])


def _placement(key: str, leaf, n_bl: int) -> NamedSharding:
    """Where a model array lives: split over baselines or device rows, else whole."""
    name = key.rsplit("/", 1)[-1]
    if np.ndim(leaf) >= 1:
        if name.startswith(DEVICE_ROWS_PREFIX) and np.shape(leaf)[0] == jax.device_count():
            return baseline_sharding()
        if (
            name in BASELINE_AXIS_NAMES
            and np.shape(leaf)[0] == n_bl
            and baselines_divide(n_bl)
        ):
            return baseline_sharding()
    return replicated_sharding()


def shard_pytree(tree: dict, n_bl: int) -> dict:
    """Place a flat dict of model arrays on the mesh.

    Per-baseline arrays (:data:`BASELINE_AXIS_NAMES`) and per-device index rows
    (:data:`DEVICE_ROWS_PREFIX`) are split along their leading axis; everything
    else is replicated. Parameters, state and constants share the layout, so
    the gradients and the optimiser's buffers follow it too.

    Identity when sharding is off. Leaves that already carry the requested
    sharding (e.g. placeholders created by :func:`replicated_zeros`) are passed
    through untouched.
    """
    if not sharding_enabled():
        return tree

    out = {}
    for key, leaf in tree.items():
        want = _placement(key, leaf, n_bl)
        if isinstance(leaf, jax.Array) and leaf.sharding == want:
            out[key] = leaf
        else:
            out[key] = make_global(leaf, want)
    return out


def replicate_off_mesh(tree):
    """Replicate every leaf of ``tree`` that is not yet placed on the mesh.

    Leaves already on :func:`mesh` keep their sharding. The rest -- the scalars
    a fresh optax state creates on one device, its step count among them -- are
    replicated, which is how a jitted step returns them. Without this the
    second step sees inputs placed differently from the first and compiles
    again. Identity when sharding is off.
    """
    if not sharding_enabled():
        return tree
    the_mesh = mesh()

    def place(leaf):
        if isinstance(leaf, jax.Array) and getattr(leaf.sharding, "mesh", None) == the_mesh:
            return leaf
        return make_global(leaf, replicated_sharding())

    return jax.tree_util.tree_map(place, tree)


def constrain_prediction(pred: dict) -> dict:
    """Replicate one-sample ``Predictive`` output inside traced code.

    Every site is written out by process 0, which can only read an array it
    holds whole, so the visibilities are gathered here, once, inside the
    compiled prediction rather than by each writer. No-op when sharding is off.
    """
    if not sharding_enabled():
        return pred
    return {
        key: lax.with_sharding_constraint(leaf, replicated_sharding())
        for key, leaf in pred.items()
    }


def replicated_zeros(shape: tuple, dtype) -> jax.Array:
    """Zeros on every device of the mesh, without a single-device copy first.

    Used for the large per-source state placeholders (``rfi_A``, ``rfi_phase``):
    every device holds them whole under baseline sharding, and building them
    through ``jnp.zeros`` would first allocate a full extra copy on device 0.
    Plain ``jnp.zeros`` when sharding is off.
    """
    if not sharding_enabled():
        return jnp.zeros(shape, dtype=dtype)
    # Canonicalize like jnp.zeros would (None -> default float; complex -> complex64
    # under f32): numpy would otherwise hand the callback f64/c128 buffers that
    # disagree with the array dtype jax expects when x64 is off.
    dtype = jnp.zeros((), dtype=dtype).dtype
    return jax.make_array_from_callback(
        tuple(shape), replicated_sharding(), lambda idx: np.zeros(_index_shape(shape, idx), dtype=dtype)
    )


def _index_shape(shape, idx) -> tuple:
    """Shape of ``zeros(shape)[idx]`` without materializing the full array."""
    return tuple(
        len(range(*s.indices(dim))) for s, dim in zip(idx, shape)
    )


def constrain_baseline_state(state: dict, n_bl: int) -> dict:
    """Pin every per-baseline state entry to the baseline sharding in traced code.

    The RFI visibility comes out of :func:`map_over_baselines` already divided,
    but the sky model and the gains are ordinary JAX built from parameters that
    are partly replicated, so XLA has no reason of its own to divide what they
    produce: it can compute a whole ``(n_bl, n_freq, n_time)`` array on every
    device and reshard it to add the two together. Saying where these belong,
    between components, keeps the whole forward on one layout. No-op unless
    the baselines divide over a sharded run.
    """
    if not baselines_divide(n_bl):
        return state
    sharding = baseline_sharding()
    out = dict(state)
    for key, leaf in state.items():
        name = key.rsplit("/", 1)[-1]
        if name in BASELINE_AXIS_NAMES and np.ndim(leaf) >= 1 and leaf.shape[0] == n_bl:
            out[key] = lax.with_sharding_constraint(leaf, sharding)
    return out


def constrain_baseline_axis(x, axis: int):
    """Pin an array whose baseline axis is not the leading one.

    The likelihood stacks the real and imaginary parts before comparing them,
    which puts the baseline axis second, out of reach of a constraint written
    for the leading one. Everything downstream of that stack -- the
    log-probability, the mask, and the cotangents of all three -- is then free
    to be computed whole on every device. No-op unless that axis divides over a
    sharded run.
    """
    if not baselines_divide(x.shape[axis]):
        return x
    spec = [None] * x.ndim
    spec[axis] = BASELINE_AXIS
    return lax.with_sharding_constraint(x, NamedSharding(mesh(), P(*spec)))


def map_over_baselines(local_fn: Callable, in_specs, out_specs=P(BASELINE_AXIS)) -> Callable:
    """Map a per-device function over the baseline mesh, with no reduction.

    ``in_specs`` says which arguments are split: ``P("bl")`` for the per-device
    index rows (see :func:`split_baselines`), ``P()`` for the per-source and
    per-antenna inputs every device needs whole. Each device computes the rows
    of its own block, and the blocks concatenate into the result -- nothing is
    summed across devices. ``shard_map`` is also what lets the FFI custom calls
    run at all, since GSPMD cannot partition a custom call.

    Reverse mode does reduce: the cotangent of a replicated input is the sum of
    every device's share, which ``shard_map`` forms with a ``psum`` in the
    transpose. That requires varying-axis checking to be off, which the
    ``ri_kernels`` primitives insist on: they do not track the varying axes, and
    with checking on the transpose would skip that ``psum`` and return wrong
    gradients.

    Unsharded it is ``local_fn`` itself, keeping the single-device path free of
    any mapping.
    """
    if not sharding_enabled():
        return local_fn

    kwargs = dict(mesh=mesh(), in_specs=in_specs, out_specs=out_specs)
    try:
        return shard_map(local_fn, check_vma=False, **kwargs)
    except TypeError:  # pragma: no cover - jax < 0.7 spells it check_rep
        return shard_map(local_fn, check_rep=False, **kwargs)


@dataclass(frozen=True)
class DeviceSplit:
    """A list of baselines divided between the devices that own their rows.

    Device ``d`` owns rows ``[d * block, (d + 1) * block)`` of the visibility
    axis. ``members[d]`` indexes the entries of the list that fall in that
    block, in list order, and ``positions[d]`` gives their rows within it,
    padded up to ``width`` -- the most any one device owns -- with ``block``:
    a spare row past the end of the block, which a padded entry is written to
    and then dropped with. Every device then computes the same number of
    baselines, which is what ``shard_map`` requires.
    """

    block: int
    width: int
    members: tuple
    positions: np.ndarray

    @property
    def padded(self) -> bool:
        """Whether any device computes a padded entry."""
        return any(len(m) < self.width for m in self.members)


def baseline_block(n_bl: int, n_dev: int | None = None) -> int:
    """Rows of the visibility axis each device owns; the last may own fewer."""
    n_dev = jax.device_count() if n_dev is None else n_dev
    return -(-n_bl // n_dev)


def split_baselines(rows, n_bl: int, n_dev: int | None = None) -> DeviceSplit:
    """Divide the baselines ``rows`` (indices into ``n_bl``) by the device owning each.

    ``rows`` is the whole visibility axis for a component that computes every
    baseline at once, or one group of it for a component that computes its
    baselines in groups (``rfi_vis:RiemannVisVariableFine``). One device when
    sharding is off, so the same plan serves both.
    """
    n_dev = (jax.device_count() if sharding_enabled() else 1) if n_dev is None else n_dev
    rows = np.asarray(rows, dtype=np.int64)
    block = baseline_block(n_bl, n_dev)
    owner = rows // block
    members = tuple(np.flatnonzero(owner == d) for d in range(n_dev))
    width = max(len(m) for m in members)
    positions = np.full((n_dev, width), block, dtype=np.int32)
    for d, m in enumerate(members):
        positions[d, : len(m)] = rows[m] - d * block
    return DeviceSplit(block, width, members, positions)


# ---------------------------------------------------------------------------
# Multi-process coordination
# ---------------------------------------------------------------------------

def barrier(name: str) -> None:
    """Block until every process reaches this point; no-op single-process."""
    if jax.process_count() == 1:
        return
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices(name)


def broadcast_bytes_from_rank0(payload, name: str) -> bytes:
    """Broadcast process 0's *payload* bytes to every process, verbatim.

    Used to share a result that only process 0 may compute -- the resolved TLE set
    -- so workers never repeat the work. Sharing the *result* rather than merely
    ordering the processes is what makes the outcome coherent when the shared
    filesystem cache cannot be written: there is then nothing on disk for a worker
    to read, and re-deriving it would mean one provider request per process.

    ``broadcast_one_to_all`` requires identical shapes on every process, so the
    length is broadcast first and the payload second. Non-zero processes pass
    ``None``. Single-process: the payload straight back.
    """
    if jax.process_count() == 1:
        return bytes(payload or b"")

    from jax.experimental import multihost_utils

    rank0 = is_process_0()
    size = np.asarray(
        multihost_utils.broadcast_one_to_all(
            np.array([len(payload) if rank0 else 0], dtype=np.int64)
        )
    )
    buffer = np.zeros(int(size[0]), dtype=np.uint8)
    if rank0 and buffer.size:
        buffer[:] = np.frombuffer(payload, dtype=np.uint8)
    # ``broadcast_one_to_all`` is implemented as a sum over the process axis.
    # JAX widens uint8 reductions (for example to uint32), so serialising the
    # result's backing storage directly would insert padding NULs between every
    # payload byte: ``b'{"'`` becomes ``b'{\0\0\0"\0\0\0'``.  Convert the
    # values back to bytes explicitly before exposing their representation.
    received = multihost_utils.broadcast_one_to_all(buffer)
    return np.asarray(received, dtype=np.uint8).tobytes()


class RankZeroFailure(RuntimeError):
    """Raised on the workers when process 0 failed at work only it does."""


def fail_together(failure, what: str) -> None:
    """End every process when the one doing the writing could not finish.

    Work only process 0 does can fail on process 0 alone, and the workers cannot
    see it: they took no part in it and are already on their way into the next
    collective -- the solve after the initial prediction is written, the chi^2
    after the optimised one, the end-of-run barrier. Process 0 unwinding there
    does not release them, it strands them in that collective until the
    coordinator times out, with nothing in any log to say why.

    Waking them is not enough either, which is what separates this from
    :func:`rank0_first`: there the workers go on to fail by themselves on the
    resource process 0 could not put in place, and here there is nothing for
    them to fail on -- the next collective simply waits.

    So the *fact* of the failure is broadcast and every process raises: process
    0 with the error it actually hit, so the run reports its real cause, and the
    workers with a :class:`RankZeroFailure` naming what was being done. The run
    then exits non-zero on every process instead of hanging on all but one.

    Call it on **every** process, unconditionally, at a point every process
    reaches whether or not anything went wrong -- it is itself a collective, and
    a process that skipped it on the happy path would strand the rest just as
    surely as one that raised past it. Single-process: the failure, straight
    back up.
    """

    if jax.process_count() == 1:
        if failure is not None:
            raise failure

        return

    from jax.experimental import multihost_utils

    # Before the raise below, never after it: a process that raised first would
    # leave every other one waiting in a broadcast it never joined.
    rank0_failed = bool(
        np.asarray(
            multihost_utils.broadcast_one_to_all(
                np.array([failure is not None], dtype=np.int32)
            )
        ).reshape(-1)[0]
    )

    if failure is not None:
        raise failure

    if rank0_failed:
        raise RankZeroFailure(
            f"process 0 failed to {what}. Stopping this process too: it is not "
            "the one that failed, but carrying on would leave it waiting in a "
            "collective process 0 will never reach. See process 0 for the cause."
        )


@contextmanager
def rank0_first(name: str):
    """Run the block on process 0 first, then on all other processes.

    Serializes shared-resource setup so workers find the resource already in place.
    Single-process: plain yield.

    Not used for the TLE fetch: sharing only the *ordering* leaves workers to
    re-derive the result from the cache, which is exactly what fails when the cache
    is unwritable. :func:`broadcast_bytes_from_rank0` shares the result itself.

    Process 0 releases the barrier from a ``finally``, so if its block raises (e.g. a
    network failure) the workers are woken instead of blocking until the coordinator
    timeout; they then fail independently on the missing resource. Turns a
    multi-process hang into a fast, symmetric error.
    """
    if jax.process_count() == 1:
        yield
        return
    if is_process_0():
        try:
            yield
        finally:
            barrier(f"rank0_first:{name}")
    else:
        barrier(f"rank0_first:{name}")
        yield


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
