import numpy as np
import cupy as cp
from nbody_streams.fields import compute_nbody_forces_gpu, compute_nbody_forces_cpu

# Minimal test case
np.random.seed(42)
N = 100
pos = np.random.randn(N, 3).astype(np.float64)
mass = np.ones(N, dtype=np.float64)
h = 0.01

print("Testing float64 GPU vs CPU forces")
print("="*60)

# CPU (reference)
acc_cpu = compute_nbody_forces_cpu(
    pos, mass, softening=h,
    kernel='spline', precision='float64'
)

# GPU with skip_validation=False
pos_gpu = cp.asarray(pos)
mass_gpu = cp.asarray(mass)

print("\n" + "="*60)
print("VALIDATED PATH (should work)")
print("="*60)
acc_gpu_validated = compute_nbody_forces_gpu(
    pos_gpu, mass_gpu, softening=h,
    kernel='spline', precision='float64',
    return_cupy=True, skip_validation=False
).get()

# GPU with skip_validation=True
h_gpu = cp.full(N, h, dtype=cp.float64)

print("\n" + "="*60)
print("FAST PATH (broken - uses strided views)")
print("="*60)

# Show the problem
print("\nMemory layout check:")
print(f"  pos_gpu.shape: {pos_gpu.shape}")
print(f"  pos_gpu.strides: {pos_gpu.strides}")
print(f"  pos_gpu[:, 0].shape: {pos_gpu[:, 0].shape}")
print(f"  pos_gpu[:, 0].strides: {pos_gpu[:, 0].strides}")
print(f"  pos_gpu[:, 0].flags.c_contiguous: {pos_gpu[:, 0].flags.c_contiguous}")

acc_gpu_fast = compute_nbody_forces_gpu(
    pos_gpu, mass_gpu, h_gpu,
    kernel='spline', precision='float64',
    return_cupy=True, skip_validation=True
).get()

print(f"\nParticle 0 positions (should all be same):")
print(f"  Original:  {pos[0]}")
print(f"  GPU array: {pos_gpu.get()[0]}")

print(f"\nParticle 0 accelerations:")
print(f"  CPU:              {acc_cpu[0]}")
print(f"  GPU (validated):  {acc_gpu_validated[0]}")
print(f"  GPU (fast path):  {acc_gpu_fast[0]}")

print(f"\nDifferences:")
diff_validated = np.linalg.norm(acc_cpu[0] - acc_gpu_validated[0])
diff_fast = np.linalg.norm(acc_cpu[0] - acc_gpu_fast[0])
print(f"  CPU vs GPU validated: {diff_validated:.2e}")
print(f"  CPU vs GPU fast:      {diff_fast:.2e}")

if diff_validated < 1e-12:
    print("\n✓ VALIDATED PATH IS CORRECT")
else:
    print("\n✗ VALIDATED PATH IS WRONG")

if diff_fast < 1e-12:
    print("✓ FAST PATH IS CORRECT")
else:
    print("✗ FAST PATH IS WRONG")
    print("\nThe bug: pos[:, 0] creates a STRIDED view, not contiguous!")
    print("The kernel reads wrong memory locations.")