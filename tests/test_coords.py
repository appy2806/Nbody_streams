"""Tests for nbody_streams.coords — coordinate and vector transforms."""

import numpy as np
import pytest
from nbody_streams.coords import (
    convert_coords,
    convert_vectors,
    convert_to_vel_los,
    generate_stream_coords,
)


# =====================================================================
# Helpers
# =====================================================================
ATOL = 1e-14  # absolute tolerance for round-trip checks


def _random_positions(shape=(200, 3), scale=10.0, rng=None):
    """Generate random Cartesian positions away from the origin."""
    rng = rng or np.random.default_rng(42)
    pos = rng.standard_normal(shape) * scale
    return pos


# =====================================================================
# convert_coords — round-trip tests
# =====================================================================
class TestConvertCoords:

    def test_cart_sph_roundtrip(self):
        pos = _random_positions()
        sph = convert_coords(pos, "cart", "sph")
        back = convert_coords(sph, "sph", "cart")
        np.testing.assert_allclose(back, pos, atol=ATOL)

    def test_cart_cyl_roundtrip(self):
        pos = _random_positions()
        cyl = convert_coords(pos, "cart", "cyl")
        back = convert_coords(cyl, "cyl", "cart")
        np.testing.assert_allclose(back, pos, atol=ATOL)

    def test_sph_cyl_roundtrip(self):
        pos = _random_positions()
        sph = convert_coords(pos, "cart", "sph")
        cyl = convert_coords(sph, "sph", "cyl")
        back = convert_coords(cyl, "cyl", "sph")
        np.testing.assert_allclose(back, sph, atol=ATOL)

    def test_identity_returns_copy(self):
        pos = _random_positions()
        out = convert_coords(pos, "cart", "cart")
        np.testing.assert_array_equal(out, pos)
        assert out is not pos  # must be a copy

    def test_batch_dims(self):
        """Arbitrary leading batch dimensions should work."""
        pos = _random_positions(shape=(4, 50, 3))
        sph = convert_coords(pos, "cart", "sph")
        assert sph.shape == (4, 50, 3)
        back = convert_coords(sph, "sph", "cart")
        np.testing.assert_allclose(back, pos, atol=ATOL)

    def test_single_point(self):
        pos = np.array([1.0, 2.0, 3.0])
        sph = convert_coords(pos, "cart", "sph")
        back = convert_coords(sph, "sph", "cart")
        np.testing.assert_allclose(back, pos, atol=ATOL)

    def test_mollweide_phi_range(self):
        """With mollweide=True, phi should be in (-pi, pi]."""
        pos = _random_positions(shape=(500, 3))
        sph = convert_coords(pos, "cart", "sph", mollweide=True)
        phi = sph[..., 2]
        assert np.all(phi > -np.pi)
        assert np.all(phi <= np.pi)

    def test_mollweide_roundtrip(self):
        pos = _random_positions()
        sph = convert_coords(pos, "cart", "sph", mollweide=True)
        back = convert_coords(sph, "sph", "cart", mollweide=True)
        np.testing.assert_allclose(back, pos, atol=ATOL)

    def test_known_values_cart_to_sph(self):
        """Point on +x axis: rho=1, theta=pi/2, phi=0."""
        pos = np.array([[1.0, 0.0, 0.0]])
        sph = convert_coords(pos, "cart", "sph")
        np.testing.assert_allclose(sph[0, 0], 1.0, atol=ATOL)           # rho
        np.testing.assert_allclose(sph[0, 1], np.pi / 2, atol=ATOL)     # theta
        np.testing.assert_allclose(sph[0, 2], 0.0, atol=ATOL)           # phi

    def test_known_values_cart_to_cyl(self):
        """Point on +x axis: R=1, phi=0, z=0."""
        pos = np.array([[1.0, 0.0, 0.0]])
        cyl = convert_coords(pos, "cart", "cyl")
        np.testing.assert_allclose(cyl[0, 0], 1.0, atol=ATOL)   # R
        np.testing.assert_allclose(cyl[0, 1], 0.0, atol=ATOL)   # phi
        np.testing.assert_allclose(cyl[0, 2], 0.0, atol=ATOL)   # z

    def test_nan_propagation(self):
        pos = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        sph = convert_coords(pos, "cart", "sph")
        assert np.all(np.isnan(sph[0]))
        assert not np.any(np.isnan(sph[1]))

    def test_invalid_system_raises(self):
        pos = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="from_sys"):
            convert_coords(pos, "xyz", "sph")
        with pytest.raises(ValueError, match="to_sys"):
            convert_coords(pos, "cart", "xyz")

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            convert_coords(np.array([1.0, 2.0]), "cart", "sph")


# =====================================================================
# convert_vectors — round-trip tests
# =====================================================================
class TestConvertVectors:

    def test_cart_sph_roundtrip(self):
        pos = _random_positions()
        vec = _random_positions(scale=1.0)
        pos_sph, vec_sph = convert_vectors(pos, vec, "cart", "sph")
        pos_back, vec_back = convert_vectors(pos_sph, vec_sph, "sph", "cart")
        np.testing.assert_allclose(vec_back, vec, atol=ATOL)
        np.testing.assert_allclose(pos_back, pos, atol=ATOL)

    def test_cart_cyl_roundtrip(self):
        pos = _random_positions()
        vec = _random_positions(scale=1.0)
        pos_cyl, vec_cyl = convert_vectors(pos, vec, "cart", "cyl")
        pos_back, vec_back = convert_vectors(pos_cyl, vec_cyl, "cyl", "cart")
        np.testing.assert_allclose(vec_back, vec, atol=ATOL)

    def test_sph_cyl_roundtrip(self):
        """sph->cyl->sph chains through cart; verify it still round-trips."""
        pos_cart = _random_positions()
        vec_cart = _random_positions(scale=1.0)
        pos_sph, vec_sph = convert_vectors(pos_cart, vec_cart, "cart", "sph")
        pos_cyl, vec_cyl = convert_vectors(pos_sph, vec_sph, "sph", "cyl")
        pos_back, vec_back = convert_vectors(pos_cyl, vec_cyl, "cyl", "sph")
        np.testing.assert_allclose(vec_back, vec_sph, atol=1e-13)

    def test_vector_magnitude_preserved(self):
        """Rotation must preserve vector magnitudes."""
        pos = _random_positions()
        vec = _random_positions(scale=5.0)
        mag_orig = np.linalg.norm(vec, axis=-1)

        _, vec_sph = convert_vectors(pos, vec, "cart", "sph")
        mag_sph = np.linalg.norm(vec_sph, axis=-1)
        np.testing.assert_allclose(mag_sph, mag_orig, atol=ATOL)

        _, vec_cyl = convert_vectors(pos, vec, "cart", "cyl")
        mag_cyl = np.linalg.norm(vec_cyl, axis=-1)
        np.testing.assert_allclose(mag_cyl, mag_orig, atol=ATOL)

    def test_batch_dims(self):
        pos = _random_positions(shape=(3, 100, 3))
        vec = _random_positions(shape=(3, 100, 3), scale=1.0)
        pos_sph, vec_sph = convert_vectors(pos, vec, "cart", "sph")
        assert pos_sph.shape == (3, 100, 3)
        assert vec_sph.shape == (3, 100, 3)

    def test_known_radial_vector(self):
        """A purely radial velocity on +x axis should map to v_rho only."""
        pos = np.array([[5.0, 0.0, 0.0]])
        vec = np.array([[3.0, 0.0, 0.0]])  # radial along +x
        _, vec_sph = convert_vectors(pos, vec, "cart", "sph")
        np.testing.assert_allclose(vec_sph[0, 0], 3.0, atol=ATOL)   # v_rho
        np.testing.assert_allclose(vec_sph[0, 1], 0.0, atol=ATOL)   # v_theta
        np.testing.assert_allclose(vec_sph[0, 2], 0.0, atol=ATOL)   # v_phi


# =====================================================================
# convert_to_vel_los
# =====================================================================
class TestConvertToVelLos:

    def test_tangential_orbit(self):
        """Circular orbit at solar position: v_los should be 0."""
        xv = np.array([8.0, 0.0, 0.0, 0.0, 220.0, 0.0])
        assert convert_to_vel_los(xv) == 0.0

    def test_radial_orbit(self):
        """Purely radial velocity along +x: v_los = vx."""
        xv = np.array([5.0, 0.0, 0.0, 100.0, 0.0, 0.0])
        np.testing.assert_allclose(convert_to_vel_los(xv), 100.0, atol=ATOL)

    def test_batch(self):
        xv = np.random.default_rng(0).standard_normal((50, 6))
        xv[:, :3] *= 10
        v_los = convert_to_vel_los(xv)
        assert v_los.shape == (50,)

    def test_3d_batch(self):
        xv = np.random.default_rng(0).standard_normal((3, 50, 6))
        xv[..., :3] *= 10
        v_los = convert_to_vel_los(xv)
        assert v_los.shape == (3, 50)

    def test_with_reference(self):
        # Particle at (10,0,0) with v=(150,0,0); progenitor at (8,0,0) v=(100,0,0)
        # Relative: pos=(2,0,0), vel=(50,0,0) -> v_los = 50
        xv = np.array([10.0, 0.0, 0.0, 150.0, 0.0, 0.0])
        ref = np.array([8.0, 0.0, 0.0, 100.0, 0.0, 0.0])
        np.testing.assert_allclose(convert_to_vel_los(xv, ref), 50.0, atol=ATOL)

    def test_zero_position_raises(self):
        xv = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="zero magnitude"):
            convert_to_vel_los(xv)


# =====================================================================
# generate_stream_coords
# =====================================================================
class TestGenerateStreamCoords:

    def _make_stream(self, rng=None):
        """Circular orbit particles spread along a stream-like arc."""
        rng = rng or np.random.default_rng(99)
        N = 100
        theta = np.linspace(-0.3, 0.3, N)  # arc along orbit
        r = 20.0
        xv = np.zeros((N, 6))
        xv[:, 0] = r * np.cos(theta)
        xv[:, 1] = r * np.sin(theta)
        xv[:, 3] = -220 * np.sin(theta)
        xv[:, 4] = 220 * np.cos(theta)
        # Add small scatter
        xv[:, :3] += rng.standard_normal((N, 3)) * 0.1
        return xv

    def test_single_stream_shape(self):
        xv = self._make_stream()
        phi1, phi2 = generate_stream_coords(xv)
        assert phi1.shape == (100,)
        assert phi2.shape == (100,)

    def test_batch_shape(self):
        xv = np.stack([self._make_stream(np.random.default_rng(i)) for i in range(3)])
        phi1, phi2 = generate_stream_coords(xv)
        assert phi1.shape == (3, 100)

    def test_progenitor_none_vs_empty(self):
        """None and [] should give identical results."""
        xv = self._make_stream()
        p1_none, p2_none = generate_stream_coords(xv, xv_prog=None)
        p1_list, p2_list = generate_stream_coords(xv, xv_prog=[])
        p1_arr, p2_arr = generate_stream_coords(xv, xv_prog=np.array([]))
        np.testing.assert_array_equal(p1_none, p1_list)
        np.testing.assert_array_equal(p1_none, p1_arr)

    def test_explicit_progenitor(self):
        xv = self._make_stream()
        phi1, phi2 = generate_stream_coords(xv, xv_prog=xv[50])
        assert phi1.shape == (100,)

    def test_degrees_vs_radians(self):
        xv = self._make_stream()
        p1_deg, _ = generate_stream_coords(xv, degrees=True)
        p1_rad, _ = generate_stream_coords(xv, degrees=False)
        np.testing.assert_allclose(np.radians(p1_deg), p1_rad, atol=1e-12)

    def test_optimizer_fit(self):
        xv = self._make_stream()
        phi1, phi2 = generate_stream_coords(xv, optimizer_fit=True)
        assert phi1.shape == (100,)
        # After optimisation, phi2 scatter should be small
        assert np.std(phi2) < np.std(phi1)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            generate_stream_coords(np.ones((10,)))


# =====================================================================
# Standalone runner
# =====================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
