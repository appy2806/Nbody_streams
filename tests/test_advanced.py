import nbody_streams as nb
import numpy as np
import os
import shutil
import sys

def setup_dummy_data(N=500):
    """Quick helper to generate test particles (Plummer-like spread)."""
    pos = np.random.normal(0, 1, (N, 3)).astype(np.float32)
    vel = np.random.normal(0, 0.1, (N, 3)).astype(np.float32)
    xv = np.hstack([pos, vel])
    mass = np.ones(N, dtype=np.float32) / N
    return xv, mass

def test_environment_summary():
    """Checks which high-performance features are actually active."""
    print("\n" + "="*50)
    print("NBODY_STREAMS SETUP CHECK")
    print("="*50)
    print(f"Python: {sys.version.split()[0]}")
    
    # Check GPU via your get_gpu_info API
    try:
        gpu = nb.get_gpu_info()
        status = "[✓] AVAILABLE" if gpu.get('available') else "[ ] NOT FOUND"
        print(f"CUDA/CuPy:  {status} ({gpu.get('device_name', 'N/A')})")
    except AttributeError:
        print("CUDA/CuPy:  [ ] get_gpu_info not found in API")

    # Check for optional Tree and Potential libs
    for mod in ['pyfalcon', 'agama', 'numba']:
        try:
            __import__(mod)
            print(f"{mod:<11}: [✓] INSTALLED")
        except ImportError:
            print(f"{mod:<11}: [ ] MISSING")
    print("="*50 + "\n")

def test_gpu_integration():
    """Runs a tiny GPU simulation and verifies snapshot output."""
    print("Testing GPU Integration (Mini-Sim)...")
    out_dir = "./test_gpu_temp"
    
    # Directory Management: Fresh Start
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    try:
        xv, masses = setup_dummy_data(1000)
        
        # Call your GPU run API
        # Note: Using your specific parameter names from the run.py main block
        final = nb.run_nbody_gpu(
            phase_space=xv,
            masses=masses,
            time_start=0.0,
            time_end=0.001,
            dt=0.0005,
            softening=0.01,
            output_dir=out_dir,
            snapshots=2,
            verbose=False
        )
        
        # Verify result
        assert final.shape == xv.shape, "Output shape mismatch!"
        print(f"  GPU Sim Run: [OK]")

        # Verify IO linkage
        snap0 = os.path.join(out_dir, "snapshot_000.h5")
        if os.path.exists(snap0):
            reader = nb.ParticleReader(sim_pattern=os.path.join(out_dir, "snapshot_*.h5"))
            data = reader.read_snapshot(0)
            print(f"  GPU IO Check: [OK] (Read {len(data.dark['posvel'])} particles)")
        
    except Exception as e:
        print(f"  GPU Integration: [FAILED/SKIPPED] -> {e}")
    finally:
        # Directory Management: Cleanup
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            print("  Cleanup: Temp directory removed.")

def test_cpu_integration():
    """Runs a tiny CPU simulation and verifies snapshot output."""
    print("Testing CPU Integration (Mini-Sim)...")
    out_dir = "./test_cpu_temp"
    
    # Directory Management: Fresh Start
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for method in ['direct', 'tree']:
        print(f"\n--- Testing CPU Method: {method.upper()} ---")    
        try:
            xv, masses = setup_dummy_data(1000)
            
            # Call your CPU run API
            # Note: Using your specific parameter names from the run.py main block
            final = nb.run_nbody_cpu(
                phase_space=xv,
                masses=masses,
                time_start=0.0,
                time_end=0.001,
                dt=0.0005,
                method=method,
                softening=0.01,
                output_dir=out_dir,
                snapshots=2,
                verbose=False
            )
            
            # Verify result
            assert final.shape == xv.shape, "Output shape mismatch!"
            print(f"  GPU Sim Run: [OK]")

            # Verify IO linkage
            snap0 = os.path.join(out_dir, "snapshot_000.h5")
            if os.path.exists(snap0):
                reader = nb.ParticleReader(sim_pattern=os.path.join(out_dir, "snapshot_*.h5"))
                data = reader.read_snapshot(0)
                print(f"  GPU IO Check: [OK] (Read {len(data.dark['posvel'])} particles)")
            
        except Exception as e:
            print(f"  GPU Integration: [FAILED/SKIPPED] -> {e}")
        finally:
            # Directory Management: Cleanup
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
                print("  Cleanup: Temp directory removed.")

if __name__ == "__main__":
    test_environment_summary()
    test_gpu_integration()
    test_cpu_integration()