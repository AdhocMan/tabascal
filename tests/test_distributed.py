"""Unit tests for the baseline-axis sharding helpers in tabascal.distributed.

Two regimes:

- In-process: the test session runs on a single CPU device, where every sharding
  helper must be an exact no-op / passthrough. This is the regime all other unit
  tests and single-GPU production runs live in, so identity here is load-bearing.
- Subprocess: multi-device behaviour needs ``XLA_FLAGS=--xla_force_host_platform_
  device_count=N`` set *before* jax initializes, so those checks run a small script
  in a fresh interpreter and assert inside it.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tabascal import distributed as dist


# ---------------------------------------------------------------------------
# Single-device (in-process): everything must be a no-op
# ---------------------------------------------------------------------------

def test_sharding_disabled_single_device():
    assert jax.device_count() == 1
    assert not dist.sharding_enabled()


def test_baselines_never_divide_single_device():
    for n in (1, 3, 32):
        assert not dist.baselines_divide(n)


def test_shard_pytree_identity_single_device():
    tree = {"vis_rfi": jnp.zeros((3, 2)), "other": jnp.ones(4)}
    assert dist.shard_pytree(tree, 3) is tree


def test_replicated_zeros_single_device():
    z = dist.replicated_zeros((3, 2), complex)
    assert z.shape == (3, 2)
    assert z.dtype == jnp.zeros((1,), dtype=complex).dtype
    np.testing.assert_array_equal(np.asarray(z), 0)


def test_constraints_identity_single_device():
    state = {"vis_rfi": jnp.zeros((4, 2))}
    assert dist.constrain_baseline_state(state, 4) is state
    x = jnp.zeros((2, 4, 3))
    assert dist.constrain_baseline_axis(x, 1) is x
    pred = {"vis_obs": jnp.zeros((1, 4))}
    assert dist.constrain_prediction(pred) is pred


def test_map_over_baselines_passthrough_single_device():
    def local_vis(a, p):
        return jnp.sum(a * p, axis=0)

    assert dist.map_over_baselines(local_vis, in_specs=()) is local_vis


def test_split_baselines_single_device_is_the_whole_list():
    split = dist.split_baselines(np.arange(6), 6)
    assert (split.block, split.width, split.padded) == (6, 6, False)
    np.testing.assert_array_equal(split.positions, [np.arange(6)])


def test_barrier_and_rank0_first_single_process():
    dist.barrier("test")  # must not raise or block
    ran = []
    with dist.rank0_first("test"):
        ran.append(True)
    assert ran == [True]


def test_is_process_0_single_process():
    assert dist.is_process_0()
    dist.print0("", end="")  # rank-0 print; must not raise


def test_process_count_single_process():
    assert dist.process_count() == 1


def test_broadcast_bytes_single_process_is_the_payload():
    assert dist.broadcast_bytes_from_rank0(b"resolved-tles", "tle-fetch") == b"resolved-tles"
    assert dist.broadcast_bytes_from_rank0(None, "tle-fetch") == b""


def test_broadcast_bytes_casts_a_widened_collective_result(monkeypatch):
    """JAX implements broadcast as a sum, which widens uint8 reductions."""
    payload = b'{"ok":true}'
    calls = 0

    def broadcast(value):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.asarray([len(payload)], dtype=np.int64)
        # This is the dtype returned when JAX reduces the uint8 payload. Calling
        # .tobytes() on it directly would insert three NULs after every byte.
        return np.frombuffer(payload, dtype=np.uint8).astype(np.uint32)

    monkeypatch.setattr(dist.jax, "process_count", lambda: 2)
    monkeypatch.setattr(dist, "is_process_0", lambda: True)
    from jax.experimental import multihost_utils
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", broadcast)

    assert dist.broadcast_bytes_from_rank0(payload, "tle-fetch") == payload


# ---------------------------------------------------------------------------
# Name/shape matching rules
# ---------------------------------------------------------------------------

def test_split_baselines_pads_every_device_to_the_widest():
    """Device d owns rows [d*block, (d+1)*block); fewer than the widest are padded."""
    # 10 baselines over 4 devices: blocks of 3, the last device owning row 9 only.
    split = dist.split_baselines(np.arange(10), 10, n_dev=4)
    assert (split.block, split.width, split.padded) == (3, 3, True)
    assert [m.tolist() for m in split.members] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    np.testing.assert_array_equal(
        split.positions, [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 3, 3]]
    )

    # A group scattered over the axis: members index the group, positions the block.
    split = dist.split_baselines([1, 2, 7, 8, 9], 10, n_dev=4)
    assert (split.block, split.width) == (3, 2)
    assert [m.tolist() for m in split.members] == [[0, 1], [], [2, 3], [4]]
    np.testing.assert_array_equal(
        split.positions, [[1, 2], [3, 3], [1, 2], [0, 3]]
    )


# ---------------------------------------------------------------------------
# Multi-device behaviour (subprocess with 4 fake CPU devices)
# ---------------------------------------------------------------------------

_MULTI_DEVICE_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import jax

    jax.config.update("jax_enable_x64", True)  # keep f64 host arrays exact on device

    import jax.numpy as jnp
    from types import SimpleNamespace
    from jax.sharding import PartitionSpec as P

    from tabascal import distributed as dist

    assert jax.device_count() == 4
    assert dist.sharding_enabled()

    # divisibility: only an even split shards the visibility-shaped arrays
    assert dist.baselines_divide(8) and not dist.baselines_divide(10)

    # shard_pytree: per-baseline leaves split, device rows split, others replicated
    rng = np.random.default_rng(0)
    tree = {
        "vis_rfi": rng.normal(size=(8, 2, 3)),
        "ast_k_r_base": rng.normal(size=(8, 3)),
        "_c/RiemannVisFine/dev_a1": np.arange(12, dtype=np.int32).reshape(4, 3),
        "rfi_A": rng.normal(size=(8, 2)),            # not a baseline name
        "vis_obs": rng.normal(size=(5, 2)),          # wrong leading dim
        "_c/ConstGains/amp_basis": rng.normal(size=(4, 3)),
    }
    sharded = dist.shard_pytree(tree, 8)
    for key, val in tree.items():
        np.testing.assert_array_equal(np.asarray(sharded[key]), val)
    assert sharded["vis_rfi"].sharding.spec == P("bl")
    assert sharded["ast_k_r_base"].sharding.spec == P("bl")
    assert sharded["_c/RiemannVisFine/dev_a1"].sharding.spec == P("bl")
    for key in ("rfi_A", "vis_obs", "_c/ConstGains/amp_basis"):
        assert sharded[key].sharding.spec == P(), key

    # an indivisible baseline count leaves the visibilities whole
    whole = dist.shard_pytree({"vis_rfi": rng.normal(size=(10, 2))}, 10)
    assert whole["vis_rfi"].sharding.spec == P()

    # pre-placed leaves pass through untouched
    again = dist.shard_pytree(sharded, 8)
    assert again["vis_rfi"] is sharded["vis_rfi"]

    # replicated_zeros: correct global shape/value, whole on every device
    z = dist.replicated_zeros((3, 5), complex)
    assert z.sharding.spec == P()
    np.testing.assert_array_equal(np.asarray(z), 0)
    assert all(s.data.shape == (3, 5) for s in z.addressable_shards)

    # constrain_baseline_state / constrain_baseline_axis inside jit
    def f(state):
        state = {"vis_rfi": state["vis_rfi"] * 2.0}
        return dist.constrain_baseline_state(state, 8)

    out = jax.jit(f)({"vis_rfi": dist.make_global(tree["vis_rfi"], dist.replicated_sharding())})
    assert out["vis_rfi"].sharding.spec == P("bl")
    x = jax.jit(lambda v: dist.constrain_baseline_axis(jnp.stack([v, v]), 1))(sharded["vis_rfi"])
    assert x.sharding.spec == P(None, "bl")

    # map_over_baselines: each device its own rows, the replicated input's
    # gradient summed over every device's share
    rows = dist.make_global(np.arange(8).reshape(4, 2), dist.baseline_sharding())
    w = jnp.arange(8.0)

    def local(r, w):
        return w[r[0]] ** 2

    mapped = dist.map_over_baselines(local, in_specs=(P("bl"), P()))
    np.testing.assert_allclose(np.asarray(mapped(rows, w)), np.arange(8.0) ** 2)
    g = jax.jit(jax.grad(lambda w: mapped(rows, w).sum()))(w)
    np.testing.assert_allclose(np.asarray(g), 2 * np.arange(8.0))

    # A fresh optimiser state is placed as _map_step returns it, and dual_run's
    # second phase only changes a state leaf, so the fit compiles its step once.
    import numpyro
    import numpyro.distributions as nd
    from tabascal import tab_tools

    def model(obs_data=None, state=None, constants=None):
        x = numpyro.sample("ast_k_r_base", nd.Normal(jnp.zeros((8, 3)), 1.0))
        numpyro.sample("obs", nd.Normal(x.sum(1), 1.0), obs=obs_data)

    params = dist.shard_pytree({"ast_k_r_base": np.ones((8, 3))}, 8)
    obs = dist.make_global(np.zeros(8), dist.baseline_sharding())
    tab_tools.run_custom_svi(model, obs, max_iter=2, init_params=params, epsilon=0.1,
                             dual_run=True, state={}, constants={})
    assert tab_tools._map_step._cache_size() == 1

    # Predictions are gathered whole for the writers
    pred = jax.jit(dist.constrain_prediction)({"vis_obs": jnp.ones((1, 8, 2))})
    assert pred["vis_obs"].sharding.spec == P()

    # Every RFI visibility component, against its own single-device run: value
    # and the gradient of every differentiated input. n_ant=5 gives 10
    # baselines, which do not divide over 4 devices (padded, visibilities
    # whole); n_ant=8 gives 28, which do (split, no padding).
    from tabascal.components.rfi_vis import (
        AnalyticVis, RiemannVisFine, RiemannVisVariableFine,
    )

    sharded_enabled = dist.sharding_enabled

    def evaluate(comp_cls, cfg, tree, grad_keys, sharded):
        dist.sharding_enabled = sharded_enabled if sharded else (lambda: False)
        try:
            comp = comp_cls()
            comp.setup(cfg)
            forward = comp.build_forward()
            tree = {**tree, **{f"{comp.prefix}/{k}": v for k, v in comp.build_constants().items()}}
            if sharded:
                tree = dist.shard_pytree(tree, cfg.n_bl)
            weights = np.arange(cfg.n_bl * cfg.n_freq * cfg.n_time, dtype=float).reshape(
                cfg.n_bl, cfg.n_freq, cfg.n_time
            )

            def loss(diff, tree):
                state = {**tree, **diff, "vis_rfi": jnp.zeros((cfg.n_bl, cfg.n_freq, cfg.n_time), complex)}
                vis = forward({}, state, tree)["vis_rfi"]
                return jnp.sum(vis.real * weights + vis.imag)

            diff = {k: tree[k] for k in grad_keys}
            return jax.jit(jax.value_and_grad(loss))(diff, tree)
        finally:
            dist.sharding_enabled = sharded_enabled

    def check(comp_cls, cfg, tree, grad_keys, label):
        got = evaluate(comp_cls, cfg, tree, grad_keys, True)
        expect = evaluate(comp_cls, cfg, tree, grad_keys, False)
        np.testing.assert_allclose(got[0], expect[0], rtol=1e-10, err_msg=label)
        for key in grad_keys:
            np.testing.assert_allclose(
                np.asarray(got[1][key]), np.asarray(expect[1][key]),
                rtol=1e-10, atol=1e-12, err_msg=f"{label}: d/d{key}",
            )

    n_rfi, n_freq, n_time, n_int_freq, n_int_time = 2, 2, 3, 2, 4
    for n_ant in (5, 8):
        a1, a2 = (np.asarray(x, dtype=np.int32) for x in np.triu_indices(n_ant, 1))
        n_bl = len(a1)
        base = dict(n_ant=n_ant, n_bl=n_bl, n_freq=n_freq, n_time=n_time, n_rfi=n_rfi,
                    n_int_freq=n_int_freq, n_int_time=n_int_time, a1=a1, a2=a2)
        fine = (n_rfi, n_ant, n_freq * n_int_freq, n_time * n_int_time)
        fine_tree = {
            "rfi_A": rng.normal(size=fine) + 1j * rng.normal(size=fine),
            "rfi_phase": rng.uniform(0, 6, fine),
        }
        # Two stride groups, interleaved over the axis, so every device's share
        # of each differs and some are padded.
        groups = [np.arange(0, n_bl, 2), np.arange(1, n_bl, 2)]
        for backend in ("ffi", "jax"):
            rfi = {"backend": backend, "baseline_block_size": 3}
            cfg = SimpleNamespace(**base, args={"rfi": rfi})
            check(RiemannVisFine, cfg, fine_tree, ("rfi_A", "rfi_phase"),
                  f"RiemannVisFine {backend} n_ant={n_ant}")
            cfg = SimpleNamespace(**base, args={"rfi": rfi},
                                  time_sample_idxs=groups, time_strides=[1, 2])
            check(RiemannVisVariableFine, cfg, fine_tree, ("rfi_A", "rfi_phase"),
                  f"RiemannVisVariableFine {backend} n_ant={n_ant}")

        # AnalyticVis under an elevation mask, with its distinct-pair padding
        mask = rng.uniform(size=(n_rfi, n_time)) > 0.3
        cfg = SimpleNamespace(
            **base, int_time=2.0, chan_width=1e6, freqs=1.4e9 + 1e6 * np.arange(n_freq),
            rfi_mask=mask,
            args={"rfi": {"analytic": {"stencil": 1, "segments": 2, "terms": 6,
                                       "cubic_terms": 3, "scratch_mb": 256}}, "data": {}},
        )
        shape = (n_rfi, n_ant, n_freq, n_time)
        tree = {
            "rfi_A": rng.normal(size=shape) + 1j * rng.normal(size=shape),
            "rfi_phase": rng.uniform(0, 6, shape),
            "rfi_delay_poly_us": rng.normal(size=(n_rfi, n_ant, n_time, 4)) * [1, 1e-4, 1e-6, 1e-8],
        }
        check(AnalyticVis, cfg, tree, ("rfi_A", "rfi_phase", "rfi_delay_poly_us"),
              f"AnalyticVis n_ant={n_ant}")

    print("MULTI_DEVICE_OK")
    """
)


def test_multi_device_helpers_subprocess():
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", _MULTI_DEVICE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "MULTI_DEVICE_OK" in result.stdout


# ---------------------------------------------------------------------------
# rank0_first control flow: process 0 must release the barrier even on error
# ---------------------------------------------------------------------------
#
# The real cross-process behaviour needs working multi-host collectives (see
# test_pipeline_multiprocess), which are unavailable in this single-host CPU test
# session. These patch rank0_first into its multi-process branch and assert the
# barrier ordering directly -- in particular that a raise inside process 0's block
# still hits the barrier, which is what wakes the workers instead of deadlocking
# them at their entry barrier until the coordinator timeout.


def _patch_multiprocess(monkeypatch, *, is_rank0: bool):
    """Force rank0_first's multi-process branch and record barrier calls."""
    calls = []
    monkeypatch.setattr(dist.jax, "process_count", lambda: 2)
    monkeypatch.setattr(dist, "is_process_0", lambda: is_rank0)
    monkeypatch.setattr(dist, "barrier", lambda name: calls.append(name))
    return calls


def test_rank0_first_releases_barrier_on_error(monkeypatch):
    calls = _patch_multiprocess(monkeypatch, is_rank0=True)

    with pytest.raises(RuntimeError):
        with dist.rank0_first("boom"):
            raise RuntimeError("process 0 failed inside the block")

    # The finally released the barrier despite the raise; workers are not stranded.
    assert calls == ["rank0_first:boom"]


def test_rank0_first_process0_barrier_after_success(monkeypatch):
    calls = _patch_multiprocess(monkeypatch, is_rank0=True)

    order = []
    with dist.rank0_first("ok"):
        order.append("body")
        assert calls == []  # barrier fires only after the block completes
    order.append("after")

    assert order == ["body", "after"]
    assert calls == ["rank0_first:ok"]


def test_rank0_first_worker_waits_before_body(monkeypatch):
    calls = _patch_multiprocess(monkeypatch, is_rank0=False)

    order = []
    with dist.rank0_first("w"):
        # Worker barriered on entry, so the resource is already in place.
        assert calls == ["rank0_first:w"]
        order.append("body")

    assert order == ["body"]


# ---------------------------------------------------------------------------
# fail_together: work only process 0 does must not end only process 0
# ---------------------------------------------------------------------------
#
# Same constraint as the rank0_first tests above -- real multi-host collectives
# are unavailable in this session -- so these force the multi-process branch and
# stand in for the broadcast. What they pin is the decision each process takes
# from what the broadcast tells it, which is what keeps a worker out of a
# collective process 0 has already unwound past.
#
# Waking the workers is *not* enough here, which is why this exists beside
# rank0_first: a woken worker carries on into the next model evaluation, and
# there is no missing resource for it to fail on -- it simply waits.


def _patch_broadcast(monkeypatch, *, rank0_failed: bool):
    """Force the multi-process branch; the broadcast reports rank 0's verdict."""

    from jax.experimental import multihost_utils

    sent = []

    def broadcast(value):
        sent.append(np.asarray(value).tolist())
        return np.array([1 if rank0_failed else 0], dtype=np.int32)

    monkeypatch.setattr(dist.jax, "process_count", lambda: 2)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", broadcast)

    return sent


class TestFailTogether:

    def test_single_process_just_raises_what_it_was_given(self):
        with pytest.raises(OSError, match="no space"):
            dist.fail_together(OSError("no space left"), "write the results")

        assert dist.fail_together(None, "write the results") is None

    def test_a_worker_stops_when_process_0_failed(self, monkeypatch):
        """The whole point: this process had no error of its own to find."""

        _patch_broadcast(monkeypatch, rank0_failed=True)

        with pytest.raises(dist.RankZeroFailure, match="write the results"):
            dist.fail_together(None, "write the results")

    def test_a_worker_carries_on_when_process_0_did_not(self, monkeypatch):
        _patch_broadcast(monkeypatch, rank0_failed=False)

        assert dist.fail_together(None, "write the results") is None

    def test_process_0_raises_the_error_it_actually_hit(self, monkeypatch):
        """Not a RankZeroFailure: the run's report has to name the real cause."""

        sent = _patch_broadcast(monkeypatch, rank0_failed=True)

        with pytest.raises(OSError, match="no space"):
            dist.fail_together(OSError("no space left"), "write the results")

        # And it told the others before raising -- a process that raised first
        # would leave every other one waiting in the broadcast it never made.
        assert sent == [[1]]

    def test_the_others_are_told_even_when_nothing_failed(self, monkeypatch):
        """The collective is unconditional, or the two sides cannot meet at all."""

        sent = _patch_broadcast(monkeypatch, rank0_failed=False)

        dist.fail_together(None, "write the results")

        assert sent == [[0]]


# ---------------------------------------------------------------------------
# Resolution broadcast: elements must be identical on every rank
# ---------------------------------------------------------------------------

class TestResolutionWireRoundTrip:
    """The single guard on the wire-format hazard.

    Process 0 resolves the satellites and broadcasts the accepted records; every
    other rank rebuilds the element frame from what it receives. A TLE survives
    that hop trivially — the two lines are text, and every rank re-parses the
    same 69 characters with the same parser. An OMM record has no lines, so the
    element *values* themselves make the trip, and a lossy projection would
    leave the ranks holding subtly different trajectory priors. Nothing would
    raise; the run would just be wrong. Hence exact equality below, never
    ``approx``.
    """

    def _round_trip(self, record, obs_epoch_jd):
        import json

        from tabascal import orbit as tle

        resolution = tle.TLEResolution([int(record["NORAD_CAT_ID"])], obs_epoch_jd, 3)
        tle._accept_remote(
            {int(record["NORAD_CAT_ID"]): dict(record)},
            "test",
            obs_epoch_jd,
            3,
            resolution.resolved,
            resolution.rejected,
        )
        assert resolution.complete, "the fixture record was not accepted"
        wire = json.loads(json.dumps(tle._resolution_to_wire(resolution)))
        return resolution, tle._resolution_from_wire(wire)

    @pytest.mark.parametrize("kind", ["tle", "omm"])
    def test_elements_survive_the_broadcast_exactly(self, kind):
        from .tle_helpers import jd, make_record

        obs = jd(2026, 8, 1)
        record = make_record(kind, 25544, obs)
        rank0, worker = self._round_trip(record, obs)

        # The wire deliberately carries a subset of the columns — workers need
        # what the elements are derived from, not the provenance. What must not
        # differ is any number the model is built from.
        before, after = rank0.frame(), worker.frame()
        for column in (
            "SEMIMAJOR_AXIS",
            "ECCENTRICITY",
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
            "MEAN_MOTION",
            "BSTAR",
            "EPOCH_JD",
        ):
            assert before[column].tolist() == after[column].tolist(), (
                f"{kind}: {column} diverged between rank 0 and the worker"
            )

    @pytest.mark.parametrize("kind", ["tle", "omm"])
    def test_epochs_and_offsets_survive_the_broadcast_exactly(self, kind):
        from .tle_helpers import jd, make_record

        obs = jd(2026, 8, 1)
        rank0, worker = self._round_trip(make_record(kind, 25544, obs), obs)

        assert worker.obs_epoch_jd == rank0.obs_epoch_jd
        assert worker.resolved[25544].epoch_jd == rank0.resolved[25544].epoch_jd
        assert worker.resolved[25544].offset_days == rank0.resolved[25544].offset_days

    def test_omm_elements_cross_the_wire_as_numbers_not_strings(self):
        # A string projection would round-trip today and mislead the next reader
        # into extending it to a field where it does not. Pin the types.
        from tabascal import orbit as tle

        from .tle_helpers import jd, make_omm

        wired = tle._wire_record(make_omm(25544, jd(2026, 8, 1)))
        assert wired["RECORD_KIND"] == "omm"
        for column in (
            "INCLINATION",
            "RA_OF_ASC_NODE",
            "ECCENTRICITY",
            "ARG_OF_PERICENTER",
            "MEAN_ANOMALY",
            "MEAN_MOTION",
            "BSTAR",
        ):
            assert isinstance(wired[column], float), column
        assert isinstance(wired["EPOCH"], str)
        assert "TLE_LINE1" not in wired

    def test_tle_projection_is_unchanged(self):
        # The TLE path predates this and must not have been disturbed.
        from tabascal import orbit as tle

        from .tle_helpers import jd, make_tle_record

        record = make_tle_record(25544, jd(2026, 8, 1))
        wired = tle._wire_record(record)
        assert wired["RECORD_KIND"] == "tle"
        assert wired["TLE_LINE1"] == record["TLE_LINE1"]
        assert wired["TLE_LINE2"] == record["TLE_LINE2"]
        assert wired["NORAD_CAT_ID"] == 25544
        assert not any(column.startswith("MEAN_") for column in wired)

    def test_a_worst_case_float_survives_the_hop(self):
        # repr() is the shortest round-tripping representation in Python 3, but
        # only if the value goes through the float path. Prove it with values
        # whose decimal expansion is long.
        import json

        from tabascal import orbit as tle

        from .tle_helpers import jd, make_omm

        awkward = {
            "INCLINATION": 51.641600000000004,
            "MEAN_MOTION": 15.721253910000001,
            "ECCENTRICITY": 0.0006703000000000001,
        }
        record = make_omm(25544, jd(2026, 8, 1), **awkward)
        wired = json.loads(json.dumps(tle._wire_record(record)))
        for column, value in awkward.items():
            assert wired[column] == value


class TestResolveSharedFailure:
    """Process 0 must always enter the collective, however it failed.

    Every other rank is already blocked in ``broadcast_one_to_all`` waiting for
    it. A failure that escapes before the broadcast does not fail the run — it
    hangs it, until the coordinator times out.
    """

    def _rank0_of_two(self, monkeypatch):
        monkeypatch.setattr(dist, "process_count", lambda: 2)
        monkeypatch.setattr(dist, "is_process_0", lambda: True)
        sent = []

        def broadcast(payload, label):
            sent.append(payload)
            return payload

        monkeypatch.setattr(dist, "broadcast_bytes_from_rank0", broadcast)
        return sent

    def test_a_resolution_failure_is_broadcast_not_raised_early(self, monkeypatch):
        import json

        from tabascal import orbit as tle

        sent = self._rank0_of_two(monkeypatch)

        def resolve():
            raise tle.TLEError("no coverage")

        with pytest.raises(tle.TLEError, match="no coverage"):
            tle.resolve_shared(resolve)
        assert sent, "process 0 skipped the broadcast every other rank waits on"
        assert json.loads(sent[0])["ok"] is False

    def test_a_serialisation_failure_is_broadcast_too(self, monkeypatch):
        import json

        from tabascal import orbit as tle

        sent = self._rank0_of_two(monkeypatch)
        # A value that resolves fine and then cannot be encoded: the failure
        # lands on json.dumps, which must be inside the guard with everything
        # else, not after it.
        monkeypatch.setattr(
            tle, "_resolution_to_wire", lambda resolution: {"ok": True, "x": object()}
        )

        with pytest.raises(tle.TLEError, match="TypeError"):
            tle.resolve_shared(lambda: None)
        assert sent, "process 0 skipped the broadcast every other rank waits on"
        assert json.loads(sent[0])["ok"] is False
