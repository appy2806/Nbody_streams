import nbody_streams as nb
import numpy as np

def test_imports():
    print("Checking package version...")
    print(f"  Package Version: {nb.__version__}")
    assert nb.__version__ != "0.0.0", "Version fallback triggered improperly."

    print("\nChecking Public API access...")
    # Test if we can see the main classes/functions
    expected_attrs = [
        'ParticleReader', 
        'run_nbody_cpu', 
        'compute_nbody_forces_cpu', 
        'G_DEFAULT'
    ]
    for attr in expected_attrs:
        has_it = hasattr(nb, attr)
        print(f"  Access to nb.{attr:<25}: {'[OK]' if has_it else '[FAILED]'}")
        assert has_it, f"Could not find {attr} in top-level namespace"

def test_linkage_smoke():
    print("\nRunning CPU linkage smoke test (N=2)...")
    # This ensures fields.py can talk to its helpers without crashing
    pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    vel = np.zeros_like(pos)
    mass = np.array([1.0, 1.0], dtype=np.float32)
    
    try:
        # Just a quick call to ensure imports inside fields.py work
        acc = nb.compute_nbody_forces_cpu(pos, mass, softening=0.1)
        print("  CPU Force Calculation: [OK]")
        assert acc.shape == (2, 3)
    except Exception as e:
        print(f"  CPU Force Calculation: [FAILED] -> {e}")
        raise

def test_privacy():
    print("\nChecking Privacy (Encapsulation)...")
    # We want to make sure the internal helpers AREN'T in the top level
    hidden_funcs = ['_save_snapshot', '_load_restart']
    for func in hidden_funcs:
        exists = hasattr(nb, func)
        print(f"  nb.{func:<26} is hidden: {'[OK]' if not exists else '[FAILED]'}")
        assert not exists, f"Internal function {func} is exposed to the user!"

if __name__ == "__main__":
    print("--- Starting nbody_streams Basic Linkage Test ---")
    try:
        test_imports()
        test_linkage_smoke()
        test_privacy()
        print("\nALL LINKAGE TESTS PASSED!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")
