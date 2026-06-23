from tabascal.imports import import_components
from tabascal.components.likelihood import gaussian
from tabascal.tab_tools import read_ms, fix_padding
from tabascal.components.trajectory import fetch_orbital_elements, get_satellite_positions
from tabascal.tle import print_spacetrack_status, preflight_tle_check
from tabascal.interferometry import (
    calculate_fringe_frequency_numpy,
    get_strides_and_idxs,
    itrf_to_uvw_numpy,
)
from tabascal.fft_gp import domain_ss
from tabascal.time import secs_to_days, mjd_to_jd, jd_to_mjd, gast_deg
from tabascal import distributed as dist

import jax.numpy as jnp

import numpy as np

import numpyro

from typing import Optional, Callable, Dict, List

from importlib.resources import files
import os
import re
import yaml
import collections.abc

    
def deep_update(d: Dict, u: Dict) -> Dict:
    """Recursively update a dictionary which includes subdictionaries.

    Parameters
    ----------
    d : Dict
        Base dictionary to update.
    u : Dict
        Update dictionary.

    Returns
    -------
    Dict
        Updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class _TabSafeLoader(yaml.SafeLoader):
    """SafeLoader whose float resolver also accepts bare scientific notation.

    PyYAML's stock resolver only treats a token as a float when the exponent is
    signed (``1.0e+9``); it parses ``1e9`` / ``3e3`` / ``209e3`` as *strings*. The
    config files use the bare form throughout, so add a resolver that accepts it.
    It is attached to this private subclass — not the shared ``yaml.SafeLoader`` —
    so importing tabascal does not reprogram YAML float parsing for the whole
    process. Anything that needs this behaviour must load via :func:`yaml_load`.
    """


_TabSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(path):
    with open(path) as f:
        return yaml.load(f, Loader=_TabSafeLoader)


def load_config(path: str) -> Dict:
    """Load a configuration file and populate default parameters where needed.

    Parameters
    ----------
    path : str
        Path to the yaml config file.
    
    Returns
    -------
    dict
        Configuration dictionary.
    """
    config_dir = files("tabascal").joinpath("data/config").__str__()
    tab_base_config_path = os.path.join(config_dir, "tab_config_base.yaml")
    base_config = yaml_load(tab_base_config_path)

    try:
        return deep_update(base_config, yaml_load(path))
    except Exception as e:
        raise IOError(f"Configuration file could not be loaded from {path}") from e

    
class TabConfig:
    """Configuration parameters for tabascal method"""

    def __init__(self, config: Dict, ms_path: str):

        # self.config = config
        self.args = config
        self.precision = config.get("model", {}).get("precision", "single")
        self.ms_path = ms_path
        self.spacetrack_path = config["satellites"].get("spacetrack_path")
        self.extra_tle_dir = config["satellites"].get("extra_tle_dir")

        print_spacetrack_status()
        preflight_tle_check(
            config["satellites"].get("norad_ids") or [],
            ms_path,
            extra_tle_dir=self.extra_tle_dir,
        )

        self.read_ms_params(
            config["data"]["freq"],
            config["data"]["corr"],
            config["data"]["data_col"],
        )
        self.set_noise(config["data"]["noise"])
        self.set_flags(config["data"]["flags"])
        config = fix_padding(
            config, self.n_freq
        )  # Bad solution, should be fixed in fft_gp. Issue when using a single frequency channel.

        self.get_orbital_elements(
            config["satellites"].get("norad_ids"),
            extra_tle_dir=config["satellites"].get("extra_tle_dir"),
        )

        config["rfi"]["min_time_bins"] = 1
        config["rfi"]["max_time_bins"] = 30

        self.n_int_time = config["rfi"]["n_int_time"]
        self.n_int_freq = config["rfi"]["n_int_freq"]

        self.estimate_rfi_sampling(
            config["rfi"]["time_int_factor"],
            config["rfi"]["min_time_bins"],
            config["rfi"]["max_time_bins"],
        )

        self._set_freqs_times()

        self._partition_baselines()

    def _partition_baselines(self):
        """Set up baseline sharding for the distributed (multi-GPU) solve.

        On a single device this is a no-op: ``mesh``/``part`` are ``None`` and the
        baseline count is unchanged. With several devices visible we pad the baseline
        axis up to a multiple of the device count (an even ``NamedSharding`` requires
        it) and bump ``n_bl`` to the padded count, which every per-baseline shape
        downstream then uses. The padding baselines are zero-valued and **flagged**
        (excluded from the likelihood, ``set_flags``/``reduced_chi2`` convention), and
        are sliced away before any result is written.
        """
        self.n_bl_true = self.n_bl  # read_ms always reports the total baseline count
        self.mesh = None
        self.part = None
        self.n_bl_local = self.n_bl

        if not dist.sharding_enabled():
            return

        # Multi-process: reuse the partition computed for the local-block read.
        # Single-process: the whole (padded) baseline axis lives on one process.
        part = self._pre_part if self._pre_part is not None else dist.bl_partition(self.n_bl)
        self.mesh = dist.make_bl_mesh()
        self.part = part

        # The per-baseline arrays currently hold this process's real baselines (a block
        # in multi-process, all of them in single-process). Pad up to local_rows -- the
        # padded rows are zero-valued and flagged out of the likelihood, and sliced away
        # before any result is written.
        n_pad = part.local_rows - self.vis_obs.shape[0]
        if n_pad:
            self.vis_obs = jnp.pad(self.vis_obs, ((0, n_pad), (0, 0), (0, 0)))
            self.flags = jnp.pad(
                self.flags, ((0, n_pad), (0, 0), (0, 0)), constant_values=True
            )
            self.uvw = jnp.pad(self.uvw, ((0, 0), (0, n_pad), (0, 0)))
            self.a1 = jnp.pad(self.a1, (0, n_pad))  # antenna 0; harmless, flagged
            self.a2 = jnp.pad(self.a2, (0, n_pad))

        self.n_bl = part.n_bl_padded  # GLOBAL padded count: every traced shape uses this
        self.n_bl_local = part.local_rows  # rows this process actually holds / builds

    def set_noise(self, noise: float):

        if noise:
            self.noise = noise

    def set_flags(self, include_flags: bool):

        if not include_flags:
            self.flags = jnp.zeros_like(self.flags, dtype=bool)

        print(f"\n{100*self.flags.mean():.1f} % Data Flagged (Not Included in Likelihood)\n")

    def read_ms_params(self, freq: float, corr: str, data_col: str):

        # Multi-process distributed solve: read only this process's baseline block so
        # host memory scales (Phase B). The partition is computed from a metadata-only
        # baseline-count probe. Single-process (incl. single-node multi-GPU) reads the
        # whole MS and shards onto devices later.
        bl_block = None
        self._pre_part = None
        import jax
        if dist.sharding_enabled() and jax.process_count() > 1:
            from tabascal.tab_tools import probe_n_bl
            n_bl_total = probe_n_bl(self.ms_path, data_col)
            self._pre_part = dist.bl_partition(n_bl_total)
            bl_block = (self._pre_part.start, self._pre_part.valid_stop)

        ms_params = read_ms(self.ms_path, freq, None, corr, data_col, bl_block=bl_block)

        self.phase_centre = {"ra": ms_params["ra"], "dec": ms_params["dec"]}
        self.dish_d = ms_params["dish_d"]
        self.ants_itrf = ms_params["ants_itrf"]
        self.vis_obs = ms_params["vis_obs"]
        self.uvw = ms_params["uvw"]
        self.flags = ms_params["flags"]

        self.n_ant = ms_params["n_ant"]
        self.n_bl = ms_params["n_bl"]
        self.n_time = ms_params["n_time"]
        self.n_freq = ms_params["n_freq"]
        self.n_corr = ms_params["n_corr"]

        self.int_time = ms_params["int_time"]
        self.times = np.asarray(ms_params["times"])
        self.times_jd = mjd_to_jd(ms_params["times_mjd"])

        self.chan_width = ms_params["chan_width"]
        self.freqs = np.asarray(ms_params["freqs"])

        self.noise = ms_params["noise"]
        self.a1 = ms_params["a1"]
        self.a2 = ms_params["a2"]

    def estimate_rfi_sampling(
        self, n_int_factor: float, min_time_bins: int, max_time_bins: int
    ):

        jd_minute = 1 / (24 * 60)
        times_jd_coarse = np.arange(
            self.times_jd[0], self.times_jd[-1] + jd_minute, jd_minute
        )
        # Satellite positions, GAST, antenna UVW and fringe frequencies are all
        # one-shot host-side setup, so always compute them in numpy/skyfield (f64):
        # faster than the jax path (no JIT compile) and accurate in both precisions.
        rfi_xyz = np.asarray(get_satellite_positions(self.tles, times_jd_coarse))

        gsa = gast_deg(times_jd_coarse)  # GAST in degrees (UTC convention)
        gh0 = (gsa - self.phase_centre["ra"]) % 360  # type: ignore

        ants_u = itrf_to_uvw_numpy(self.ants_itrf, gh0, self.phase_centre["dec"])[:, :, 0]

        get_fringe_freq = lambda rfi_pos: calculate_fringe_frequency_numpy(
            jd_to_mjd(times_jd_coarse),
            np.max(self.freqs),
            rfi_pos,
            self.ants_itrf,
            ants_u,
            self.phase_centre["dec"],
        )
        # fringe_freq is shape (n_rfi, n_time_coarse, n_bl)
        fringe_freq = np.array([get_fringe_freq(rfi_pos) for rfi_pos in rfi_xyz])

        # Global max over all processes' baseline blocks: this sets the integration
        # sample count (n_int_time), which fixes fine-grid array shapes and so must be
        # identical on every process. No-op single-process.
        self.max_rfi_vis = dist.all_max(np.max(np.abs(self.vis_obs)))
        sample_freq_bl = (
            np.pi
            * np.max(np.abs(fringe_freq), axis=(0, 1))
            * np.sqrt(self.max_rfi_vis / (6 * self.noise))
        )
        n_int_times = np.ceil(n_int_factor * self.int_time * sample_freq_bl).astype(int)

        # time_sample_idxs and time_strides are only used in RiemannVisTimeFreqVariable
        self.time_sample_idxs, self.time_strides, self.n_int_time = (
            get_strides_and_idxs(n_int_times, min_time_bins, max_time_bins)
        )

    def _set_freqs_times(self):

        ns = [self.n_freq, self.n_time]
        ss_factors = [self.n_int_freq, self.n_int_time]
        pad_factors = [
            self.args["rfi"]["freq_pad_factor"],
            self.args["rfi"]["time_pad_factor"],
        ]
        # domain_ss is jax-based, so under jax_enable_x64=False it builds the grids
        # in f32 internally. The real grids carry large magnitudes (freqs ~1e9 Hz,
        # and times_jd_fine ~2.4e6 JD) that lose all usable precision in f32. Since
        # domain_ss is affine in (x0, dx) (output = x0 + dx * normalised_grid), build
        # the normalised grid with x0=0, dx=1 (small, f32-safe) and apply the real
        # offset/scale in numpy f64.
        unit_freqs, unit_times = domain_ss(
            ns, [1.0, 1.0], [0.0, 0.0], ss_factors, pad_factors
        )
        self.freqs_fine = self.freqs[0] + self.chan_width * np.asarray(
            unit_freqs, dtype=np.float64
        )
        self.times_fine = self.times[0] + self.int_time * np.asarray(
            unit_times, dtype=np.float64
        )
        self.n_freq_fine = len(self.freqs_fine)
        self.n_time_fine = len(self.times_fine)
        self.times_jd_fine = self.times_jd[0] + secs_to_days(self.times_fine)

    def get_orbital_elements(self, norad_ids: List[int], extra_tle_dir: Optional[str] = None):

        obs_epoch_jd = float(self.times_jd.mean())

        self.elements, self.epoch_jd, self.norad_ids, self.tles = (
            fetch_orbital_elements(obs_epoch_jd, norad_ids, extra_tle_dir=extra_tle_dir)
        )
        self.n_rfi = len(self.norad_ids)


class Model:

    def __init__(
        self,
        tab_config: TabConfig,
        component_list: List[str],
        likelihood: Callable = gaussian,
    ):

        self.noise = tab_config.noise
        # Take `flags` as an explicit argument rather than closing over it. Under a
        # multi-process solve the flags array is baseline-sharded across
        # non-addressable devices, and JAX forbids closing over such arrays inside a
        # jitted step ("Closing over jax.Array that spans non-addressable devices...").
        # It is threaded through `constants` instead (see build_prob_model).
        self.likelihood = lambda pred, obs_data, flags: likelihood(
            pred, obs_data, {"noise": tab_config.noise, "flags": flags}
        )

        components = [C() for C in import_components(component_list)]
        self.components = components
        for comp in components:
            comp.setup(tab_config)

        init_params = [comp.init_params_base for comp in components]
        self.init_params = {k: v for d in init_params for k, v in d.items()}

        state = [comp.state_outputs for comp in components]
        self.state = {k: v for d in state for k, v in d.items()}

        self.constants = {}
        for comp in components:
            for key, value in comp.build_constants().items():
                self.constants[f"{comp.prefix}/{key}"] = value

        # Flags travel with `constants` so the likelihood receives them as an explicit
        # (baseline-sharded) traced argument instead of a closure capture. Added before
        # _shard_for_distributed so it gets sharded along the baseline axis like the
        # other per-baseline constants.
        self.constants["flags"] = tab_config.flags

        self.state["vis_ast"] = jnp.zeros_like(self.state["vis_obs"])
        self.state["vis_rfi"] = jnp.zeros_like(self.state["vis_obs"])

        self.state["rmse_ast"] = jnp.array([jnp.nan])
        self.state["rmse_rfi"] = jnp.array([jnp.nan])
        self.state["rmse_gains"] = jnp.array([jnp.nan])

        self.forward = self.build_forward()
        self.prob_model = self.build_prob_model()

        self._shard_for_distributed(tab_config)

    def _shard_for_distributed(self, tab_config):
        """Place every model array on the device mesh for the distributed solve.

        Per-baseline leaves (``vis_*`` state, ``ast_k_*`` params,
        ``sigma_ast_k``/``mu_ast_k`` and ``a1``/``a2`` constants) are sharded along the
        baseline axis; per-antenna params, GP kernels and scalars are replicated --
        decided purely by leading-dim == ``n_bl`` in :func:`distributed.put_leaf`. The
        observed visibilities and flags on ``tab_config`` are sharded too: ``vis_obs``
        is the optimizer's ``obs_data`` (and ``flags`` rides in ``constants``, passed as
        an explicit argument to the likelihood), so both must match the prediction's
        baseline sharding. ``tab_config.flags`` is also sharded here for the host-side
        metrics. No-op on one device (``mesh is None``).
        """
        mesh = getattr(tab_config, "mesh", None)
        if mesh is None:
            return

        # Each process contributes its own per-baseline block; shard_bl assembles the
        # global (padded) jax.Arrays. Per-antenna params / scalars are replicated.
        part = tab_config.part
        self.init_params = dist.shard_pytree(self.init_params, part, mesh)
        self.state = dist.shard_pytree(self.state, part, mesh)
        self.constants = dist.shard_pytree(self.constants, part, mesh)
        tab_config.vis_obs = dist.put_leaf(tab_config.vis_obs, part, mesh)
        tab_config.flags = dist.put_leaf(tab_config.flags, part, mesh)

    def build_forward(self):
        forwards = [comp.build_forward() for comp in self.components]

        def forward(params, state, constants):

            for sub_forward in forwards:
                state = sub_forward(params, state, constants)

            return state

        return forward

    def build_set_params(self):
        set_params_functions = [comp.build_set_params() for comp in self.components]

        def set_params():
            params = {}

            for set_params in set_params_functions:
                params = set_params(params)

            return params

        return set_params

    def build_prob_model(self):

        set_params = self.build_set_params()
        forward = self.forward
        likelihood = self.likelihood

        def prob_model(obs_data=None, state=None, constants=None):

            params = set_params()

            state = forward(params, state, constants)

            numpyro.deterministic("rfi_phase", state["rfi_phase"])
            numpyro.deterministic("rfi_A", state["rfi_A"])

            numpyro.deterministic("vis_rfi", state["vis_rfi"])
            numpyro.deterministic("vis_ast", state["vis_ast"])
            numpyro.deterministic("gains", state["gains"])
            numpyro.deterministic("vis_obs", state["vis_obs"])

            if obs_data is not None:
                likelihood(state["vis_obs"], obs_data, constants["flags"])

            return state

        return prob_model
