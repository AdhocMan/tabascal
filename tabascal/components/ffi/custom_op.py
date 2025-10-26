import jax
import ctypes
from functools import partial
import os
from jax.extend import core
from jax.interpreters import mlir, ad, xla
from jax.core import ShapedArray
import jax.numpy as jnp

dir_path = os.path.dirname(os.path.realpath(__file__))
tab_lib = ctypes.cdll.LoadLibrary(f"{dir_path}/tab.so")
jax.ffi.register_ffi_target(
    "calc_rfi", jax.ffi.pycapsule(tab_lib.calc_rfi_vis_cpu), platform="cpu")
jax.ffi.register_ffi_target(
    "calc_rfi_jvp", jax.ffi.pycapsule(tab_lib.calc_rfi_jvp_cpu), platform="cpu")
jax.ffi.register_ffi_target(
    "calc_rfi_transpose", jax.ffi.pycapsule(tab_lib.calc_rfi_transpose_cpu), platform="cpu")

tab_lib_gpu = ctypes.cdll.LoadLibrary(f"{dir_path}/tab_gpu.so")
jax.ffi.register_ffi_target(
    "calc_rfi_gpu", jax.ffi.pycapsule(tab_lib_gpu.calc_rfi_vis_gpu), platform="gpu")
jax.ffi.register_ffi_target(
    "calc_rfi_jvp_gpu", jax.ffi.pycapsule(tab_lib_gpu.calc_rfi_jvp_gpu), platform="gpu")

rfi_jvp_op = core.Primitive("rfi_jvp_op")
rfi_jvp_op.def_impl(partial(xla.apply_primitive, rfi_jvp_op))

def rfi_jvp_abstract(a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad):
    # rfi_amp_fine and rfi_phase shape is
    # (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)
    n_time = rfi_amp_fine.shape[4]
    n_freq = rfi_amp_fine.shape[2]
    n_bl = a1.shape[0]
    return ShapedArray([a1.shape[0], n_freq, n_time], rfi_amp_fine.dtype)

rfi_jvp_op.def_abstract_eval(rfi_jvp_abstract)

def rfi_jvp_lowering_cpu(ctx, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad):
    res = jax.ffi.ffi_lowering("calc_rfi_jvp")
    print("========== call custom jvp kernel= ==========")
    return [res(ctx, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad)]

mlir.register_lowering(rfi_jvp_op, rfi_jvp_lowering_cpu, platform='cpu')

def rfi_jvp_lowering_gpu(ctx, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad):
    res = jax.ffi.ffi_lowering("calc_rfi_jvp_gpu")
    print("========== call custom GPU jvp kernel= ==========")
    return [res(ctx, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad)]

mlir.register_lowering(rfi_jvp_op, rfi_jvp_lowering_gpu, platform='gpu')


def rfi_jvp_transpose(g, a1, a2, rfi_amp_fine, rfi_amp_fine_grad, rfi_phase, rfi_phase_grad):
  print("========== call custom transpose kernel= ==========")
  call = jax.ffi.ffi_call(
    "calc_rfi_transpose",
    (rfi_amp_fine, rfi_phase),
    vmap_method="sequential",
  )
  t1, t2 = call(a1, a2, rfi_amp_fine, rfi_phase, g)

  return None, None, t1, t1, t2, t2

ad.primitive_transposes[rfi_jvp_op] = rfi_jvp_transpose


rfi_vis_op = core.Primitive("rfi_vis_op")
rfi_vis_op.def_impl(partial(xla.apply_primitive, rfi_vis_op))

def rfi_vis_abstract(a1, a2, rfi_amp_fine, rfi_phase):
    # rfi_amp_fine and rfi_phase shape is
    # (n_rfi, n_ant, n_freq, n_int_freq, n_time, n_int_time)
    n_time = rfi_amp_fine.shape[4]
    n_freq = rfi_amp_fine.shape[2]
    n_bl = a1.shape[0]
    return ShapedArray([a1.shape[0], n_freq, n_time], rfi_amp_fine.dtype)

rfi_vis_op.def_abstract_eval(rfi_vis_abstract)

def rfi_vis_lowering_cpu(ctx, a1, a2, rfi_amp_fine, rfi_phase):
    res = jax.ffi.ffi_lowering("calc_rfi")
    print("========== call custom kernel= ==========")
    return [res(ctx, a1, a2, rfi_amp_fine, rfi_phase)]


mlir.register_lowering(rfi_vis_op, rfi_vis_lowering_cpu, platform='cpu')

def rfi_vis_lowering_gpu(ctx, a1, a2, rfi_amp_fine, rfi_phase):
    res = jax.ffi.ffi_lowering("calc_rfi_gpu")
    print("========== call custom GPU kernel= ==========")
    return [res(ctx, a1, a2, rfi_amp_fine, rfi_phase)]


mlir.register_lowering(rfi_vis_op, rfi_vis_lowering_gpu, platform='gpu')

def rfi_vis_jvp(args, tangents):
  a1, a2, rfi_amp_fine, rfi_phase = args
  a1_dot, a2_dot, rfi_amp_fine_dot, rfi_phase_dot = tangents

  if type(rfi_amp_fine_dot) is ad.Zero:
      rfi_amp_fine_dot = jnp.zeros(rfi_amp_fine.shape, rfi_amp_fine.dtype)
  if type(rfi_phase_dot) is ad.Zero:
      rfi_phase_dot = jnp.zeros(rfi_phase.shape, rfi_phase.dtype)


  grad = rfi_jvp_op.bind(a1, a2, rfi_amp_fine, rfi_amp_fine_dot, rfi_phase, rfi_phase_dot)
  #  grad = rfi_jvp_op.bind(a1, a2, rfi_amp_fine, rfi_phase)

  return rfi_vis_op.bind(a1, a2, rfi_amp_fine, rfi_phase), grad

ad.primitive_jvps[rfi_vis_op] = rfi_vis_jvp
