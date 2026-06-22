"""Equivalence tests for the distributed (multi-GPU) baseline-sharded solve.

These exercise the building blocks of Strategy C on **two host devices** -- the same
mesh / sharding / ``shard_map`` code paths used on real GPUs -- and assert the sharded
result matches the single-device reference in both value and gradient. Two devices
require ``--xla_force_host_platform_device_count=2``, which must be set before JAX
initialises; since the rest of the suite runs single-device, each case is launched in
a subprocess with that flag (mirroring the subprocess pattern in
``test_tabascal_pipeline.py``).

Run directly (``python tests/test_distributed.py <case>``) to debug a single case.
"""

import glob
import os
import socket
import subprocess
import sys

import pytest

CASES = ["primitives", "ffi_op", "map_step", "model_shard"]

# A real tab-sim Measurement Set for the read/assembly tests (no TLEs needed to read
# visibilities). Skipped if the example data is not present.
_MS_GLOB = os.path.join(
    os.path.dirname(__file__), os.pardir, "examples", "data", "*", "*.ms"
)


def _example_ms():
    hits = sorted(glob.glob(_MS_GLOB))
    return hits[0] if hits else None


@pytest.mark.parametrize("case", CASES)
def test_distributed_case(case):
    env = dict(os.environ)
    # Two CPU devices + true fp32 matmuls (match production precision, see conftest).
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    ).strip()
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, __file__, case], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        f"distributed case {case!r} failed:\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# The cases below run in the two-device subprocess.
# ---------------------------------------------------------------------------

def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def case_primitives():
    """shard_bl / gather_bl / replicate round-trip and produce the right specs."""
    import numpy as np
    import tabascal.distributed as d

    _check(d.sharding_enabled(), "expected >1 device")
    mesh = d.make_bl_mesh()
    part = d.bl_partition(7)  # 7 baselines, 2 devices -> block 4, padded 8
    _check(part.n_bl_padded == 8, f"padded {part.n_bl_padded}")

    full = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
    g = d.shard_bl(d.pad_bl(full, part), part, mesh)
    _check(tuple(g.sharding.spec) == (d.BL_AXIS, None), g.sharding.spec)
    _check(np.allclose(d.gather_bl(g)[:7], full), "round-trip mismatch")

    rep = d.replicate(np.ones(5, np.float32), mesh)
    _check(len(rep.sharding.spec) == 0, rep.sharding.spec)


def _toy_arrays():
    import numpy as np
    import jax.numpy as jnp

    n_ant = 8
    pairs = [(i, j) for i in range(n_ant) for j in range(i + 1, n_ant)]
    rng = np.random.default_rng(1)
    cf = lambda *s: jnp.asarray(
        rng.standard_normal(s) + 1j * rng.standard_normal(s), jnp.complex64
    )
    rf = lambda *s: jnp.asarray(rng.standard_normal(s), jnp.float32)
    a1 = jnp.asarray([p[0] for p in pairs], jnp.int32)
    a2 = jnp.asarray([p[1] for p in pairs], jnp.int32)
    n_bl, nf, nt, n_rfi, nif, nit = len(pairs), 2, 4, 1, 1, 2
    amp = cf(n_ant, nf, nt, n_rfi, nif, nit)
    ph = rf(n_ant, nf, nt, n_rfi, nif, nit)
    return n_ant, a1, a2, n_bl, nf, nt, amp, ph, cf, rf


def case_ffi_op():
    """Sharded RFI-vis FFI op matches the unsharded op in forward and gradient."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import tabascal.distributed as d
    from tabascal.components.ffi.rfi_vis_op import rfi_vis_sharded, _rfi_vis_eval

    n_ant, a1, a2, *_rest, amp, ph, _cf, _rf = _toy_arrays()
    ref = _rfi_vis_eval(n_ant, a1, a2, amp, ph)

    mesh = d.make_bl_mesh()
    bl, rep = NamedSharding(mesh, P(d.BL_AXIS)), NamedSharding(mesh, P())
    a1s, a2s = jax.device_put(a1, bl), jax.device_put(a2, bl)
    amps, phs = jax.device_put(amp, rep), jax.device_put(ph, rep)

    out = jax.jit(lambda a, b, c, e: rfi_vis_sharded(n_ant, a, b, c, e, mesh=mesh))(
        a1s, a2s, amps, phs
    )
    _check(tuple(out.sharding.spec) == (d.BL_AXIS,), out.sharding.spec)
    _check(np.allclose(np.asarray(out), np.asarray(ref), atol=1e-5), "forward mismatch")

    loss_ref = lambda amp, ph: jnp.sum(jnp.abs(_rfi_vis_eval(n_ant, a1, a2, amp, ph)) ** 2)
    loss_shd = lambda amp, ph: jnp.sum(
        jnp.abs(rfi_vis_sharded(n_ant, a1s, a2s, amp, ph, mesh=mesh)) ** 2
    )
    gr = jax.grad(loss_ref, (0, 1))(amp, ph)
    gs = jax.jit(jax.grad(loss_shd, (0, 1)))(amps, phs)
    _check(np.allclose(np.asarray(gr[0]), np.asarray(gs[0]), atol=1e-4), "grad amp")
    _check(np.allclose(np.asarray(gr[1]), np.asarray(gs[1]), atol=1e-4), "grad phase")


def case_map_step():
    """A neg-log-post mirroring the model (sharded per-bl + replicated per-ant + FFI
    op + masked Gaussian all-reduce) matches single-device in value and gradient.

    The gradient on the *replicated* per-antenna params is the key check: it must be
    all-reduced across baseline shards, which GSPMD does automatically.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    import tabascal.distributed as d
    from tabascal.components.ffi.rfi_vis_op import rfi_vis_sharded

    n_ant, a1, a2, n_bl, nf, nt, amp, ph, cf, _rf = _toy_arrays()
    ast, obs = cf(n_bl, nf, nt), cf(n_bl, nf, nt)
    rng = np.random.default_rng(2)
    flags = jnp.asarray(rng.random((n_bl, nf, nt)) < 0.1)
    noise = jnp.float32(0.5)

    def nlp(ast, amp, ph, obs, flags, a1, a2, mesh):
        vis = ast + rfi_vis_sharded(n_ant, a1, a2, amp, ph, mesh=mesh)
        sq = jnp.where(~flags, (jnp.abs(vis - obs) / noise) ** 2, 0.0)
        return jnp.sum(sq) / (2 * obs.size)

    val_ref = nlp(ast, amp, ph, obs, flags, a1, a2, None)
    g_ref = jax.grad(
        lambda a, b, c: nlp(a, b, c, obs, flags, a1, a2, None).real, (0, 1, 2)
    )(ast, amp, ph)

    mesh = d.make_bl_mesh()
    bl, rep = NamedSharding(mesh, P(d.BL_AXIS)), NamedSharding(mesh, P())
    asts, obss, fls = (jax.device_put(x, bl) for x in (ast, obs, flags))
    a1s, a2s = jax.device_put(a1, bl), jax.device_put(a2, bl)
    amps, phs = jax.device_put(amp, rep), jax.device_put(ph, rep)

    val = jax.jit(partial(nlp, mesh=mesh))(asts, amps, phs, obss, fls, a1s, a2s)
    g = jax.jit(
        lambda a, b, c: jax.grad(
            lambda x, y, z: nlp(x, y, z, obss, fls, a1s, a2s, mesh).real, (0, 1, 2)
        )(a, b, c)
    )(asts, amps, phs)

    _check(np.allclose(np.asarray(val), np.asarray(val_ref), rtol=1e-5, atol=1e-6), "value")
    _check(len(val.sharding.spec) == 0, f"loss should be replicated: {val.sharding.spec}")
    _check(np.allclose(np.asarray(g[0]), np.asarray(g_ref[0]), atol=1e-4), "grad ast (sharded)")
    _check(np.allclose(np.asarray(g[1]), np.asarray(g_ref[1]), atol=1e-4), "grad amp (replicated)")
    _check(np.allclose(np.asarray(g[2]), np.asarray(g_ref[2]), atol=1e-4), "grad phase (replicated)")


def test_read_ms_bl_block_equivalence():
    """read_ms(bl_block) over disjoint blocks reassembles the full single-shot read.

    This is the concrete memory-scaling enabler: each process reads only its baselines.
    Runs in-process (single device), so no subprocess needed.
    """
    import numpy as np
    from tabascal.tab_tools import read_ms

    ms = _example_ms()
    if ms is None:
        pytest.skip("no example MS available")

    full = read_ms(ms)
    n_bl = full["n_bl"]
    bnds = [0, n_bl // 3, 2 * (n_bl // 3), n_bl]
    for key, axis in [("vis_obs", 0), ("flags", 0), ("uvw", 1), ("a1", 0), ("a2", 0)]:
        parts = []
        for i in range(3):
            b = read_ms(ms, bl_block=(bnds[i], bnds[i + 1]))
            assert b["n_bl"] == n_bl, "n_bl must stay the total count"
            assert float(b["noise"]) == float(full["noise"]), "noise must be global"
            parts.append(np.asarray(b[key]))
        cat = np.concatenate(parts, axis=axis)
        assert np.array_equal(cat, np.asarray(full[key])), f"{key} block mismatch"


def test_multiprocess_read_assembly():
    """Two real processes (jax.distributed on CPU) each read their baseline block and
    assemble the global vis_obs via make_array_from_process_local_data; it must equal
    the single-shot full read. Exercises the multi-process Phase B backbone end-to-end.
    """
    ms = _example_ms()
    if ms is None:
        pytest.skip("no example MS available")

    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"  # one CPU device per process -> 2 global devices
    env.pop("XLA_FLAGS", None)  # do NOT force multiple devices per process
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "mp", str(i), str(port), ms],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for i in range(2)
    ]
    outs = [p.communicate(timeout=300) for p in procs]
    for i, (p, (out, err)) in enumerate(zip(procs, outs)):
        assert p.returncode == 0, f"process {i} failed:\n{out}\n{err}"


def case_mp(process_id, port, ms_path):
    import jax
    jax.distributed.initialize(
        coordinator_address=f"localhost:{port}",
        num_processes=2,
        process_id=int(process_id),
    )
    import numpy as np
    import tabascal.distributed as d
    from tabascal.tab_tools import read_ms, probe_n_bl

    _check(d.sharding_enabled() and jax.process_count() == 2, "expected 2 processes")
    n_bl = probe_n_bl(ms_path)
    part = d.bl_partition(n_bl)
    local = read_ms(ms_path, bl_block=(part.start, part.valid_stop))
    mesh = d.make_bl_mesh()
    g = d.shard_bl(d.pad_bl(np.asarray(local["vis_obs"]), part), part, mesh)
    full = d.gather_bl(g)[:n_bl]
    if d.is_process_0():
        ref = np.asarray(read_ms(ms_path)["vis_obs"])
        _check(np.array_equal(full, ref), "multiprocess assembly != full read")


def case_model_shard():
    """``shard_pytree`` shards per-baseline leaves and replicates the rest.

    Mirrors what ``Model._shard_for_distributed`` does to ``init_params`` / ``state`` /
    ``constants``: the leading-dim == n_bl rule must shard the per-baseline arrays
    (``ast_k_*``, ``vis_*``, ``a1``/``a2``, ``sigma_ast_k``) and replicate the
    per-antenna params (``rfi_*``, ``gains``), GP kernels and scalars.
    """
    import numpy as np
    import tabascal.distributed as d

    mesh = d.make_bl_mesh()
    # Realistic interferometer sizing: n_bl = n_ant*(n_ant-1)/2, distinct from n_ant /
    # n_rfi, and divisible by the 2 devices. n_ant=5 -> n_bl=10.
    n_ant, n_rfi, n_bl = 5, 1, 10
    tree = {
        # per-baseline -> sharded
        "ast_k_r_base": np.zeros((n_bl, 4, 5), np.float32),
        "vis_obs": np.zeros((n_bl, 2, 3), np.complex64),
        "a1": np.zeros((n_bl,), np.int32),
        "sigma_ast_k": np.zeros((n_bl, 4, 5), np.float32),
        # per-antenna / kernels / scalars -> replicated
        "rfi_r_induce_base": np.zeros((n_rfi, n_ant, 1, 6), np.float32),
        "gains": np.ones((n_ant, 2, 3), np.complex64),
        "L_rfi_A": np.zeros((6, 6), np.float32),
        "noise": np.float32(0.5),
    }
    part = d.bl_partition(n_bl)  # 2 devices: block 5, padded 10, local_rows 10
    out = d.shard_pytree(tree, part, mesh)

    sharded = {"ast_k_r_base", "vis_obs", "a1", "sigma_ast_k"}
    for k, v in out.items():
        spec = v.sharding.spec
        if k in sharded:
            _check(tuple(spec)[0] == d.BL_AXIS, f"{k} should be bl-sharded, got {spec}")
        else:
            _check(len(spec) == 0, f"{k} should be replicated, got {spec}")


if __name__ == "__main__":
    if sys.argv[1] == "mp":
        case_mp(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        {
            "primitives": case_primitives,
            "ffi_op": case_ffi_op,
            "map_step": case_map_step,
            "model_shard": case_model_shard,
        }[sys.argv[1]]()
    print(f"{sys.argv[1]} OK")
