from math import isfinite

import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P
import numpy as np

from tabascal.distributed import BASELINE_AXIS, map_over_baselines, split_baselines
from tabascal.interferometry import (
    calculate_rfi_vis_blocked,
    calculate_rfi_vis_variable,
)
from tabascal.components import Component
from tabascal.poly_interp import fine_offsets, interp_tables, monomial_tables
from ri_kernels.jax_api import (
    RFIVisOp,
    analytic_eval_with_indices,
    analytic_indices,
    vis_eval_with_indices,
)


#: The index arrays of ``RFIVisOp``, in the order ``vis_eval_with_indices`` takes them.
_VIS_INDICES = ("a1", "a1_sorter", "a1_start", "a2", "a2_sorter", "a2_start")
#: The index arrays of ``RFIAnalyticVisOp``, as ``analytic_indices`` returns them.
_ANALYTIC_INDICES = _VIS_INDICES + ("pair_index", "tile_pairs")
#: The index arrays the ``jax`` backends read: the baselines' own antennas.
_JAX_INDICES = ("a1", "a2")


def _unused_pairs(a1, a2, n_pad, n_ant):
    """``n_pad`` ordered antenna pairs that the list ``(a1, a2)`` does not hold.

    Padding for ``RFIAnalyticVisOp``, whose staged GPU kernels write each pair
    once: a padding pair repeating one the device already computes would leave
    that baseline unwritten. Ordered pairs are distinct from their reverses and
    autocorrelations are pairs too, so ``n_ant ** 2`` exist and a device's list,
    no longer than the baseline axis, always leaves room.
    """
    taken = set(zip(np.asarray(a1).tolist(), np.asarray(a2).tolist()))
    pairs = []
    for p in range(n_ant):
        for q in range(n_ant):
            if (p, q) not in taken:
                pairs.append((p, q))
                if len(pairs) == n_pad:
                    return np.array(pairs, dtype=np.int32).T
    raise ValueError(f"no {n_pad} antenna pairs are left to pad with")


def _device_pairs(a1, a2, rows, n_bl, n_ant, distinct=False):
    """Each device's antenna pairs for the baselines ``rows``, padded to one length.

    Returns the :class:`~tabascal.distributed.DeviceSplit` and the per-device
    ``a1`` and ``a2``, each ``(n_devices, width)`` int32. A device owning fewer
    than ``width`` of the rows is padded: with the list's first pair, which the
    ``jax`` backends and ``RFIVisOp`` compute without complaint, or with pairs
    it does not hold when ``distinct`` (see :func:`_unused_pairs`). Padded
    rows are dropped from the output, so they get a zero cotangent and add
    nothing to the gradients.
    """
    split = split_baselines(rows, n_bl)
    a1 = np.asarray(a1, dtype=np.int32)[np.asarray(rows)]
    a2 = np.asarray(a2, dtype=np.int32)[np.asarray(rows)]
    dev_a1, dev_a2 = [], []
    for members in split.members:
        p1, p2 = a1[members], a2[members]
        n_pad = split.width - len(members)
        if n_pad:
            if distinct:
                pad1, pad2 = _unused_pairs(p1, p2, n_pad, n_ant)
            else:
                pad1 = np.full(n_pad, a1[0], dtype=np.int32)
                pad2 = np.full(n_pad, a2[0], dtype=np.int32)
            p1, p2 = np.concatenate([p1, pad1]), np.concatenate([p2, pad2])
        dev_a1.append(p1)
        dev_a2.append(p2)
    return split, np.stack(dev_a1), np.stack(dev_a2)


def _stack_indices(names, per_device, suffix=""):
    """Constants ``dev_<name><suffix>``: one row of each index array per device."""
    return {
        f"dev_{name}{suffix}": jnp.stack([jnp.asarray(ix[j]) for ix in per_device])
        for j, name in enumerate(names)
    }


def _vis_indices(n_ant, dev_a1, dev_a2):
    """``RFIVisOp``'s index arrays for each device's baselines."""
    return [
        RFIVisOp(n_ant, jnp.asarray(p1), jnp.asarray(p2)).indices
        for p1, p2 in zip(dev_a1, dev_a2)
    ]


def _own_rows(rows):
    """The index rows a device was handed, without the device axis the map leaves."""
    return tuple(x[0] for x in rows)


def _read_backend(config, backends):
    """The ``rfi.backend`` a component runs, from the ones it has, default first.

    ``jax`` is the visibility written in JAX and differentiated by it; ``ffi``
    is the compiled ``ri_kernels`` op, with its own JVP and transpose. A null
    or absent key is the component's default.
    """
    backend = config.args["rfi"].get("backend")
    if backend is None:
        return backends[0]
    if backend not in backends:
        raise ValueError(
            f"rfi.backend must be one of {', '.join(map(repr, backends))} or null "
            f"for this component, got {backend!r}."
        )
    return backend


class RiemannVisFine(Component):
    """Riemann-sum RFI visibilities on the fine grid, through ``rfi.backend``.

    ``ffi``, the default, is ``ri_kernels``' compiled ``RFIVisOp``, which bounds
    the fine grid inside the kernel and reads no block size. See
    ``docs/kernels.md``.

    ``jax`` is the reference implementation, the one ``ffi`` is validated
    against in value, forward mode and reverse mode. It walks the
    baseline axis in blocks of ``rfi.baseline_block_size`` under ``checkpoint``
    (see :func:`tabascal.interferometry.calculate_rfi_vis_blocked`) so that the
    fine grid it integrates is bounded by the block rather than by the whole
    array: what the forward pass leaves behind for reverse mode is the result
    and a transposed copy of its per-antenna inputs, not the ``(n_bl, n_rfi,
    n_freq_fine, n_time_fine)`` intermediate the reduction is built from.

    It trades recomputation for memory rather than aiming at speed. The block
    size does not change the result -- baselines are independent -- only how much
    of the fine grid is live at once, and how many scan steps that takes. A null
    block size is every baseline in a single step: the fine grid is still
    recomputed rather than stored, so the tape stays small, but it is formed
    whole. Measured on one GH200 that peaks where the unscanned kernel did, and
    across four it is well under it, the tape being per-device memory that the
    collective does not divide.
    """

    # Accumulates into vis_rfi, which Model zeroes before the components run.
    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.n_int_freq
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.n_ant = config.n_ant
            self.backend = _read_backend(config, ("ffi", "jax"))

            # null is a setting, not a missing value: one block over every
            # baseline, which keeps the checkpoint and leaves the scan a single
            # step. int() alone would turn 1.9 into 1 without a word: one
            # baseline per scan step, dressed up as a valid setting. The
            # finiteness test guards int() against yaml's .inf and .nan, which
            # raise there with a message about floats rather than about the key;
            # it is asked of floats only, since an int is finite by construction
            # and float() on a big enough one raises in its turn.
            block_size = config.args["rfi"].get("baseline_block_size", 128)
            if block_size is not None and (
                isinstance(block_size, bool)
                or not isinstance(block_size, (int, float))
                or (isinstance(block_size, float) and not isfinite(block_size))
                or block_size != int(block_size)
                or block_size < 1
            ):
                raise ValueError(
                    "rfi.baseline_block_size is the number of baselines handled "
                    "per scan step: a whole number of at least 1, or null for a "
                    f"single block over every baseline, got {block_size!r}."
                )
            self.baseline_block_size = (
                None if block_size is None else int(block_size)
            )

            # Each device's own baselines. The whole axis is one list, split
            # into contiguous blocks, so a device's rows come out of the op in
            # the order of its block and need no scatter.
            _, dev_a1, dev_a2 = _device_pairs(
                self.a1, self.a2, np.arange(self.n_bl), self.n_bl, self.n_ant
            )
            if self.backend == "ffi":
                self._indices = _stack_indices(
                    _VIS_INDICES, _vis_indices(self.n_ant, dev_a1, dev_a2)
                )
            else:
                self._indices = _stack_indices(_JAX_INDICES, list(zip(dev_a1, dev_a2)))

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        return dict(self._indices)

    def build_forward(self):
        """Return pure, JIT-compatible function

        Under sharding each device computes the rows of its own block of the
        visibility axis from the whole per-source signal, and the blocks
        concatenate into the result: nothing is summed across devices. The
        rows past ``n_bl`` are the last device's padding, dropped here.
        """
        prefix = self.prefix
        n_bl = self.n_bl
        names = _VIS_INDICES if self.backend == "ffi" else _JAX_INDICES
        local_vis = self._local_vis_ffi() if self.backend == "ffi" else self._local_vis_jax()
        n_idx = len(names)

        def device_vis(*args):
            return local_vis(_own_rows(args[:n_idx]), *args[n_idx:])

        mapped = map_over_baselines(
            device_vis, in_specs=(P(BASELINE_AXIS),) * n_idx + (P(), P())
        )

        def forward(params, state, constants):
            rows = [constants[f"{prefix}/dev_{name}"] for name in names]
            vis_rfi = mapped(*rows, state["rfi_A"], state["rfi_phase"])[:n_bl]
            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _local_vis_jax(self):
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        block_size = self.baseline_block_size

        # The fine->coarse mean runs per baseline block, inside the scan, which
        # is what bounds the fine grid.
        def local_vis(indices, rfi_A, rfi_phase):
            a1, a2 = indices
            return calculate_rfi_vis_blocked(
                rfi_A, rfi_phase, a1, a2, n_int_freq, n_int_time, block_size
            )

        return local_vis

    def _local_vis_ffi(self):
        n_int_time = self.n_int_time
        n_int_freq = self.n_int_freq
        n_time = self.n_time
        n_freq = self.n_freq
        n_ant = self.n_ant

        def local_vis(indices, rfi_A, rfi_phase):
            new_shape = (-1, n_ant, n_freq, n_int_freq, n_time, n_int_time)
            rfi_amp_fine = rfi_A.reshape(new_shape)
            rfi_phase_fine = rfi_phase.reshape(new_shape)

            # Transpose to (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
            rfi_amp_fine = jnp.transpose(rfi_amp_fine, (1, 2, 4, 0, 3, 5))
            rfi_phase_fine = jnp.transpose(rfi_phase_fine, (1, 2, 4, 0, 3, 5))

            return vis_eval_with_indices(indices, rfi_amp_fine, rfi_phase_fine)

        return local_vis

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }


class RiemannVisVariableFine(Component):
    """:class:`RiemannVisFine` with each baseline group sampled at its own stride.

    The baselines are split into the groups of ``time_sample_idxs``, each read at
    every ``time_stride``-th fine time sample. ``rfi.backend`` picks ``ffi`` (the
    default), one ``RFIVisOp`` call per group, or ``jax``.

    Under sharding each device computes, group by group, the baselines of its
    own block of the visibility axis, and scatters them into that block.
    """

    # Accumulates into vis_rfi, which Model zeroes before the components run.
    required_inputs = {
        "rfi_phase": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "rfi_A": ("n_rfi", "n_ant", "n_freq_fine", "n_time_fine"),
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {"vis_rfi": ("n_bl", "n_freq", "n_time")}

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.a1 = config.a1
            self.a2 = config.a2
            self.n_int_time = config.n_int_time
            self.n_int_freq = config.n_int_freq
            self.n_rfi = config.n_rfi
            self.n_ant = config.n_ant
            self.n_time = config.n_time
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.backend = _read_backend(config, ("ffi", "jax"))

            self.time_sample_idxs = config.time_sample_idxs
            self.time_strides = config.time_strides

            self._setup_device_groups()
            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    def _setup_device_groups(self):
        """Split every group between the devices that own its baselines.

        A group's baselines are scattered over the visibility axis, so how many
        of them a device owns differs from device to device; each device's
        share is padded to the most any device holds, and its padded entries
        are scattered into a spare row past the end of its block and dropped.
        """
        names = _VIS_INDICES if self.backend == "ffi" else _JAX_INDICES
        self._groups, self._indices, self._spare = [], {}, False
        self._block = split_baselines([], self.n_bl).block
        for i, idx in enumerate(self.time_sample_idxs):
            if not len(idx):
                continue
            split, dev_a1, dev_a2 = _device_pairs(
                self.a1, self.a2, idx, self.n_bl, self.n_ant
            )
            per_device = (
                _vis_indices(self.n_ant, dev_a1, dev_a2)
                if self.backend == "ffi" else list(zip(dev_a1, dev_a2))
            )
            self._indices.update(_stack_indices(names, per_device, f"_{i}"))
            self._indices[f"dev_pos_{i}"] = jnp.asarray(split.positions)
            self._groups.append(i)
            self._spare = self._spare or split.padded

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        return dict(self._indices)

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        n_bl, n_freq, n_time = self.n_bl, self.n_freq, self.n_time
        groups = tuple(self._groups)
        strides = tuple(self.time_strides[i] for i in groups)
        names = _VIS_INDICES if self.backend == "ffi" else _JAX_INDICES
        n_idx = len(names) + 1  # and the positions
        block = self._block
        rows = block + 1 if self._spare else block
        inputs = self._inputs_ffi() if self.backend == "ffi" else self._inputs_jax()
        evaluate = self._group_vis_ffi if self.backend == "ffi" else self._group_vis_jax

        def device_vis(*args):
            rfi_A, rfi_phase = inputs(*args[-2:])
            vis_rfi = jnp.zeros((rows, n_freq, n_time), dtype=complex)
            for g, stride in enumerate(strides):
                *indices, pos = _own_rows(args[g * n_idx:(g + 1) * n_idx])
                vis_rfi = vis_rfi.at[pos].set(
                    evaluate(tuple(indices), rfi_A, rfi_phase, stride)
                )
            # The spare row only ever holds padding.
            return vis_rfi[:block]

        n_args = len(groups) * n_idx
        mapped = map_over_baselines(
            device_vis, in_specs=(P(BASELINE_AXIS),) * n_args + (P(), P())
        )

        def forward(params, state, constants):
            args = [
                constants[f"{prefix}/dev_{name}_{i}"]
                for i in groups for name in (*names, "pos")
            ]
            vis_rfi = mapped(*args, state["rfi_A"], state["rfi_phase"])[:n_bl]

            # vis_rfi is shape (n_bl, n_freq, n_time)
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _inputs_jax(self):
        shape = (-1, self.n_ant, self.n_freq, self.n_int_freq, self.n_time, self.n_int_time)

        # calculate_rfi_vis_variable expects the n_rfi axis on axis 1 and
        # reduces over it internally, so reshape to (n_rfi, n_ant, ...) and
        # swap to (n_ant, n_rfi, n_freq, n_int_freq, n_time, n_int_time).
        def inputs(rfi_A, rfi_phase):
            return tuple(jnp.swapaxes(jnp.reshape(x, shape), 0, 1) for x in (rfi_A, rfi_phase))

        return inputs

    @staticmethod
    def _group_vis_jax(indices, rfi_A, rfi_phase, time_stride):
        a1, a2 = indices
        return calculate_rfi_vis_variable(rfi_A, rfi_phase, a1, a2, 1, time_stride)

    def _inputs_ffi(self):
        shape = (-1, self.n_ant, self.n_freq, self.n_int_freq, self.n_time, self.n_int_time)

        # Transpose to (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
        def inputs(rfi_A, rfi_phase):
            return tuple(
                jnp.transpose(jnp.reshape(x, shape), (1, 2, 4, 0, 3, 5))
                for x in (rfi_A, rfi_phase)
            )

        return inputs

    @staticmethod
    def _group_vis_ffi(indices, rfi_amp_fine, rfi_phase, time_stride):
        # Subsample the integration-time axis by the group's stride, mirroring
        # calculate_rfi_vis_variable. The FFI kernel then reduces over the
        # remaining integration samples.
        t_idx = slice(time_stride // 2, None, time_stride)
        return vis_eval_with_indices(
            indices, rfi_amp_fine[..., t_idx], rfi_phase[..., t_idx]
        )

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }


#: The ``rfi.analytic`` options and their ranges; ``None`` is unbounded.
_ANALYTIC_OPTIONS = {
    "stencil": (0, 4),
    "segments": (1, 1024),
    "terms": (1, 32),
    "cubic_terms": (0, 8),
    "scratch_mb": (1, None),
}


class AnalyticVis(Component):
    """RFI visibilities from the data grid, each cell integrated in closed form.

    Reads the signal and phase of
    :class:`~tabascal.components.rfi_signal.ComplexRFIVarAnt` (or
    ``ComplexRFIConstAnt``) and :class:`~tabascal.components.trajectory.FixedOrbit`
    on the data grid, with the delay polynomial beside them, and hands them to
    ``ri_kernels.jax_api.RFIAnalyticVisOp``. Inside each cell the signal is the
    polynomial through the ``2 * rfi.analytic.stencil + 1`` nearest cells on each
    axis and the phase is linear in frequency and a Taylor series in time; the
    time integral is closed form and the frequency one the ``n_int_freq``-point
    rule of the fine grid. No fine time grid is formed, so ``n_int_time`` is
    never read.

    The kernel differentiates the signal, the phase and the delay; the interval
    and the tables are constants. With ``FixedOrbit`` the phase and the delay
    are constants too, and the signal-only kernels run. Under an elevation mask each source
    is integrated alone, over the whole axis, and its visibility masked per
    cell. See ``docs/analytic_rfi_vis.md``.
    """

    # Accumulates into vis_rfi, which Model zeroes before the components run.
    required_inputs = {
        "rfi_A": ("n_rfi", "n_ant", "n_freq", "n_time"),
        "rfi_phase": ("n_rfi", "n_ant", "n_freq", "n_time"),
        "rfi_delay_poly_us": ("n_rfi", "n_ant", "n_time", "n_path"),
        "vis_rfi": ("n_bl", "n_freq", "n_time"),
    }
    output_shapes = {"vis_rfi": ("n_bl", "n_freq", "n_time")}
    integrates_cells = True

    parameters = {}

    def setup(self, config):
        """All validation and error-prone operations here"""
        try:
            self.n_ant = config.n_ant
            self.n_bl = config.n_bl
            self.n_freq = config.n_freq
            self.n_time = config.n_time
            # Only the compiled op for now: the JAX reference lands with it in
            # ri_kernels and becomes the second backend then.
            self.backend = _read_backend(config, ("ffi",))
            self.options = self._read_options(config.args["rfi"]["analytic"])

            # The kernel writes each antenna pair once, so a repeated pair would
            # be left unwritten. A reversed pair or an autocorrelation is fine.
            pairs, counts = np.unique(
                np.stack([np.asarray(config.a1), np.asarray(config.a2)], axis=1),
                axis=0, return_counts=True,
            )
            if np.any(counts > 1):
                repeated = ", ".join(str(tuple(int(a) for a in p)) for p in pairs[counts > 1])
                raise ValueError(
                    f"each (a1, a2) baseline must appear at most once; repeated: {repeated}."
                )
            # Each device's own baselines, padded with pairs it does not hold.
            _, dev_a1, dev_a2 = _device_pairs(
                config.a1, config.a2, np.arange(self.n_bl), self.n_bl, self.n_ant,
                distinct=True,
            )
            self._indices = _stack_indices(
                _ANALYTIC_INDICES,
                [analytic_indices(self.n_ant, p1, p2) for p1, p2 in zip(dev_a1, dev_a2)],
            )

            # Host-side float64 tables, once. MHz against delays in
            # microseconds: their product is cycles.
            h = self.options.pop("stencil")
            chan_width = float(config.chan_width)
            dnu = fine_offsets(config.n_int_freq, chan_width)
            self.w_freq, self.start_freq = interp_tables(self.n_freq, h, dnu / chan_width)
            self.g_time, self.start_time = monomial_tables(self.n_time, h)
            self.dnu_mhz = dnu / 1e6
            self.freq_mhz = np.asarray(config.freqs, dtype=np.float64) / 1e6
            self.int_time = float(config.int_time)
            # The elevation mask multiplies each source's visibility after its
            # cells are interpolated, so a visible cell's stencil reads its
            # hidden neighbours' signal.
            mask = getattr(config, "rfi_mask", None)
            self.rfi_mask = None if mask is None or np.all(mask) else np.asarray(mask, bool)

            self._set_outputs()

        except Exception as e:
            raise RuntimeError(f"{self.__class__.__name__} setup failed: {e}")

    @staticmethod
    def _read_options(analytic):
        unknown = sorted(set(analytic) - set(_ANALYTIC_OPTIONS))
        if unknown:
            raise ValueError(
                f"rfi.analytic.{unknown[0]} is not an option; the options are "
                f"{', '.join(_ANALYTIC_OPTIONS)}."
            )
        options = {}
        for key, (lo, hi) in _ANALYTIC_OPTIONS.items():
            value = analytic[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < lo
                or (hi is not None and value > hi)
            ):
                bounds = f"from {lo} to {hi}" if hi is not None else f"of at least {lo}"
                raise ValueError(
                    f"rfi.analytic.{key} must be a whole number {bounds}, got {value!r}."
                )
            options[key] = value
        return options

    def build_set_params(self):

        def set_params(params):
            return params

        return set_params

    def build_constants(self):
        mask = {} if self.rfi_mask is None else {"rfi_mask": jnp.asarray(self.rfi_mask)}
        return {
            **mask,
            **self._indices,
            "w_freq": jnp.asarray(self.w_freq),
            "start_freq": jnp.asarray(self.start_freq),
            "g_time": jnp.asarray(self.g_time),
            "start_time": jnp.asarray(self.start_time),
            "dnu_mhz": jnp.asarray(self.dnu_mhz),
            "freq_mhz": jnp.asarray(self.freq_mhz),
            "int_time": jnp.asarray(self.int_time),
        }

    def build_forward(self):
        """Return pure, JIT-compatible function"""
        prefix = self.prefix
        options, n_bl = self.options, self.n_bl
        masked = self.rfi_mask is not None
        n_idx = len(_ANALYTIC_INDICES)

        def forward(params, state, constants):
            real = [
                constants[f"{prefix}/{name}"]
                for name in ("w_freq", "g_time", "dnu_mhz", "int_time", "freq_mhz")
            ]
            start_freq = constants[f"{prefix}/start_freq"].astype(jnp.int32)
            start_time = constants[f"{prefix}/start_time"].astype(jnp.int32)

            # Each device's own baselines from the whole per-source signal. The
            # kernel wants the antenna axis first and every real input in the
            # phase's dtype.
            def device_vis(*args):
                indices = _own_rows(args[:n_idx])
                rfi_A, rfi_phase, rfi_delay, *mask = args[n_idx:]
                dtype = rfi_phase.dtype
                w_freq, g_time, dnu_mhz, int_time, freq_mhz = (x.astype(dtype) for x in real)
                vis = lambda A, phase, delay: analytic_eval_with_indices(
                    indices,
                    jnp.swapaxes(A, 0, 1),
                    jnp.swapaxes(phase, 0, 1),
                    jnp.swapaxes(delay, 0, 1).astype(dtype),
                    w_freq, start_freq, g_time, start_time, dnu_mhz, int_time, freq_mhz,
                    **options,
                )
                if not mask:
                    return vis(rfi_A, rfi_phase, rfi_delay)

                # One source per step, its mask applied to its visibility.
                def step(total, source):
                    *inputs, m = source
                    return total + jnp.where(m, vis(*(x[None] for x in inputs)), 0), None

                total = jnp.zeros(
                    (indices[0].shape[0], w_freq.shape[0], rfi_A.shape[-1]), rfi_A.dtype
                )
                return lax.scan(step, total, (rfi_A, rfi_phase, rfi_delay, *mask))[0]

            rows = [constants[f"{prefix}/dev_{name}"] for name in _ANALYTIC_INDICES]
            mask = [constants[f"{prefix}/rfi_mask"]] if masked else []
            signal = (state["rfi_A"], state["rfi_phase"], state["rfi_delay_poly_us"], *mask)
            mapped = map_over_baselines(
                device_vis, in_specs=(P(BASELINE_AXIS),) * n_idx + (P(),) * len(signal)
            )
            # The rows past n_bl are the last device's padding.
            vis_rfi = mapped(*rows, *signal)[:n_bl]
            state = {**state, "vis_rfi": state["vis_rfi"] + vis_rfi}

            return state

        return forward

    def _set_outputs(self):

        self.state_outputs = {
            "vis_rfi": jnp.zeros((self.n_bl, self.n_freq, self.n_time), dtype=complex),
        }
