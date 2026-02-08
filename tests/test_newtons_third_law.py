# test_newtons_third_law.py
"""
Test Newton's 3rd Law using nbody_streams public API.
This will help diagnose if there's an issue with the package setup.
"""
import numpy as np
import nbody_streams as nbody

def test_newtons_third_law(N=1000, precision='float32', verbose=True):
    """
    Test that net force is ~0 for isolated system.
    
    Returns:
        bool: True if test passes
    """
    np.random.seed(42)
    
    # Generate test data with EXACTLY equal masses
    pos = np.random.randn(N, 3)
    mass = np.ones(N)  # All exactly 1.0
    h = 0.01
    
    # Convert to correct dtype
    if precision in ['float32', 'float32_kahan']:
        dtype = np.float32
        eps_expected = 1.2e-7  # float32 machine epsilon
        threshold = 1e-5  # Should be well below this
    else:
        dtype = np.float64
        eps_expected = 2.2e-16
        threshold = 1e-13
    
    pos = pos.astype(dtype)
    mass = mass.astype(dtype)
    
    # Compute forces
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing Newton's 3rd Law: {precision.upper()}")
        print(f"{'='*60}")
        print(f"N particles: {N}")
        print(f"All masses equal: {np.allclose(mass, mass[0])}")
        print(f"Mass value: {mass[0]}")
    
    acc = nbody.compute_nbody_forces_gpu(
        pos, mass, softening=h,
        precision=precision,
        kernel='spline',
        skip_validation=False,
        return_cupy=False
    )
    
    # Check data types
    if verbose:
        print(f"\nData types:")
        print(f"  pos.dtype: {pos.dtype}")
        print(f"  mass.dtype: {mass.dtype}")
        print(f"  acc.dtype: {acc.dtype}")
    
    # Compute net force
    # For equal masses: F_net = m * sum(a_i) = sum(F_i) should be ~0
    net_force = np.sum(acc * mass[:, np.newaxis], axis=0)
    net_mag = np.linalg.norm(net_force)
    
    # Typical force magnitude
    typical_force = np.median(np.linalg.norm(acc, axis=1) * mass)
    relative_error = net_mag / (typical_force * N)
    
    if verbose:
        print(f"\nResults:")
        print(f"  Net force: {net_force}")
        print(f"  Magnitude: {net_mag:.3e}")
        print(f"  Typical force: {typical_force:.3e}")
        print(f"  Relative error: {relative_error:.3e}")
        print(f"  Machine epsilon: {eps_expected:.3e}")
        print(f"  Error / epsilon: {relative_error / eps_expected:.1f}x")
    
    # Check if passes
    passed = net_mag < threshold
    
    if verbose:
        if passed:
            print(f"\n  ✓ PASS - Net force < {threshold:.1e}")
        else:
            print(f"\n  ✗ FAIL - Net force = {net_mag:.3e} > {threshold:.1e}")
            print(f"  Expected: ~{eps_expected:.1e}, Got: {net_mag:.3e}")
            print(f"  This is {net_mag/eps_expected:.0f}x larger than machine epsilon!")
    
    return passed, net_mag, relative_error


def test_all_precisions():
    """Test all precision modes."""
    print("\n" + "="*60)
    print("NEWTON'S 3RD LAW TEST - ALL PRECISIONS")
    print("="*60)
    
    results = {}
    for precision in ['float32', 'float32_kahan', 'float64']:
        passed, net_mag, rel_err = test_newtons_third_law(
            N=1000, 
            precision=precision,
            verbose=True
        )
        results[precision] = {
            'passed': passed,
            'net_force_magnitude': net_mag,
            'relative_error': rel_err
        }
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for prec, res in results.items():
        status = "✓ PASS" if res['passed'] else "✗ FAIL"
        print(f"{prec:15s}: {status} (net force = {res['net_force_magnitude']:.3e})")
        all_passed = all_passed and res['passed']
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nDiagnostic suggestions:")
        print("1. Clear CuPy cache: rm -rf ~/.cupy/kernel_cache/*")
        print("2. Restart Python/Jupyter kernel")
        print("3. Check that pos/mass arrays are contiguous")
        print("4. Verify editable install: pip show nbody_streams")
    
    return all_passed


if __name__ == "__main__":
    # Run tests
    all_passed = test_all_precisions()
    
    # Additional diagnostic: Check import path
    print("\n" + "="*60)
    print("DIAGNOSTIC INFO")
    print("="*60)
    print(f"nbody_streams location: {nbody.__file__}")
    print(f"compute_nbody_forces_gpu: {nbody.compute_nbody_forces_gpu}")
    
    # Check if using correct module
    import nbody_streams.fields
    print(f"fields module: {nbody_streams.fields.__file__}")
    
    # Exit code
    import sys
    sys.exit(0 if all_passed else 1)