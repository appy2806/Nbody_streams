"""Direct comparison of GPU tree-code vs nbody_streams direct-sum."""
import cupy as cp
import numpy as np
from nbody_streams.tree_gpu import tree_gravity_gpu
from nbody_streams import compute_nbody_forces_gpu, compute_nbody_potential_gpu

n = 100_000
eps = 0.05

# Simple gaussian positions centered at origin
rng = cp.random.default_rng(42)
pos = rng.standard_normal((n, 3), dtype=cp.float32) * 0.3
mass = cp.full(n, 1.0/n, dtype=cp.float32)

# Reference: direct-sum
print("Computing direct-sum reference...")
acc_ref = cp.asarray(compute_nbody_forces_gpu(
    pos, mass, softening=eps, precision='float32', kernel='plummer', G=1.0,
))
phi_ref = cp.asarray(compute_nbody_potential_gpu(
    pos, mass, softening=eps, precision='float32', kernel='plummer', G=1.0,
))
cp.cuda.runtime.deviceSynchronize()

# GPU tree
print("Computing GPU tree-code...")
acc_bon, phi_bon = tree_gravity_gpu(pos, mass, eps=eps, theta=0.5, G=1.0)
cp.cuda.runtime.deviceSynchronize()

# Compare accelerations
print(f"\nN={n}, eps={eps}, theta=0.5")
print(f"ref acc[0]:    {acc_ref[0].get()}")
print(f"tree acc[0]:   {acc_bon[0].get()}")
print(f"ref phi[0]:    {float(phi_ref[0]):.6f}")
print(f"tree phi[0]:   {float(phi_bon[0]):.6f}")

# Error statistics
diff = cp.linalg.norm(acc_bon - acc_ref, axis=1)
amag = cp.linalg.norm(acc_ref, axis=1)
rel_err = diff / amag.clip(1e-10)
print(f"\nAcceleration relative error:")
print(f"  mean:   {float(rel_err.mean()):.4e}")
print(f"  median: {float(cp.median(rel_err)):.4e}")
print(f"  max:    {float(rel_err.max()):.4e}")

phi_err = cp.abs(phi_bon - phi_ref) / cp.abs(phi_ref).clip(1e-10)
print(f"\nPotential relative error:")
print(f"  mean:   {float(phi_err.mean()):.4e}")
print(f"  median: {float(cp.median(phi_err)):.4e}")
print(f"  max:    {float(phi_err.max()):.4e}")

# Sanity: force direction
r_hat = pos / cp.linalg.norm(pos, axis=1, keepdims=True).clip(1e-10)
a_rad_ref = cp.sum(acc_ref * r_hat, axis=1)
a_rad_bon = cp.sum(acc_bon * r_hat, axis=1)
print(f"\nFraction inward (a_r < 0):")
print(f"  reference: {float((a_rad_ref < 0).sum())/n:.4f}")
print(f"  tree:      {float((a_rad_bon < 0).sum())/n:.4f}")
