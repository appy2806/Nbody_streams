"""
tests/test_snapshot_schedule.py
Regression tests: the GPU-tree backend must write the *same* snapshot schedule
as the direct-sum GPU backend.

Background
----------
``run_nbody_gpu_tree`` used to derive its schedule from
``snap_every = max(1, n_steps // snapshots)`` and save on
``current_step % snap_every == 0``.  Whenever ``snapshots`` did not divide
``n_steps`` this produced

  * more (or fewer) than ``snapshots`` datasets, and
  * a final snapshot short of ``time_end``.

Example: ``time_start=7.90065004, time_end=13.799, dt=5e-4`` -> n_steps=11797
with ``snapshots=300`` gave ``snap_every=39``, 303 datasets, and a last
snapshot at step 11778 instead of 11797.

Both backends now build ``snapshot_steps = np.round(np.linspace(0, n_steps,
snapshots))`` and drain it with the same while-loop, so ids are 0-based
``000..snapshots-1`` (ParticleReader depends on 0-based ids) and the last
snapshot always lands on ``time_end``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="both GPU backends require CuPy")

from nbody_streams.run import run_nbody_gpu
from nbody_streams.tree_gpu.run_gpu_tree import run_nbody_gpu_tree

G = 4.30092e-6  # kpc, km/s, Msun


def _plummer_ic(N: int, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    """Small cold blob — physics is irrelevant here, only the I/O schedule is."""
    rng = np.random.default_rng(seed)
    phase_space = np.empty((N, 6), dtype=np.float64)
    phase_space[:, :3] = rng.normal(scale=0.5, size=(N, 3))
    phase_space[:, 3:] = rng.normal(scale=1.0, size=(N, 3))
    masses = np.full(N, 1e4, dtype=np.float64)
    return phase_space, masses


def _read_schedule(output_dir: Path) -> tuple[list[int], list[float]]:
    """Return (sorted snapshot ids, snap_time per id) across all snapshot files."""
    ids: list[int] = []
    times: dict[int, float] = {}
    for f in sorted(output_dir.glob("snapshot*.h5")):
        with h5py.File(f, "r") as h5:
            if "snapshots" not in h5:
                continue
            grp = h5["snapshots"]
            for name in grp:
                if not name.startswith("snap."):
                    continue
                idx = int(name.split(".")[1])
                ids.append(idx)
                times[idx] = float(grp.attrs[f"snap_time.{idx:03d}"])
    ids.sort()
    return ids, [times[i] for i in ids]


def _run_both(time_start: float, time_end: float, dt: float, snapshots: int,
              N: int = 2048) -> tuple[tuple[list[int], list[float]],
                                      tuple[list[int], list[float]]]:
    phase_space, masses = _plummer_ic(N)
    common = dict(
        time_start=time_start,
        time_end=time_end,
        dt=dt,
        softening=0.05,
        G=G,
        external_potential=None,
        snapshots=snapshots,
        save_snapshots=True,
        restart_interval=10 ** 9,   # keep restart I/O out of the way
        verbose=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        direct_dir = Path(tmp) / "direct"
        tree_dir = Path(tmp) / "tree"

        run_nbody_gpu(phase_space.copy(), masses, output_dir=str(direct_dir), **common)
        run_nbody_gpu_tree(phase_space.copy(), masses, output_dir=str(tree_dir), **common)

        return _read_schedule(direct_dir), _read_schedule(tree_dir)


# n_steps for each case: 11797 (300 does not divide it — the case that exposed
# the bug), 60 (1 snapshot), 50 (7 does not divide it).
@pytest.mark.parametrize(
    "time_start, time_end, dt, snapshots",
    [
        pytest.param(7.90065004, 13.799, 5e-4, 300,
                     marks=pytest.mark.slow, id="11797steps-300snaps"),
        pytest.param(0.0, 0.03, 5e-4, 1, id="60steps-1snap"),
        pytest.param(0.0, 0.025, 5e-4, 7, id="50steps-7snaps"),
    ],
)
def test_tree_matches_direct_snapshot_schedule(time_start, time_end, dt, snapshots):
    (ids_direct, t_direct), (ids_tree, t_tree) = _run_both(
        time_start, time_end, dt, snapshots
    )

    n_steps = int(round((time_end - time_start) / dt))

    # Exact count, 0-based contiguous ids (ParticleReader relies on 0-based).
    assert ids_direct == list(range(snapshots)), (
        f"direct backend wrote ids {ids_direct[:5]}..{ids_direct[-5:]}, "
        f"expected 0..{snapshots - 1}"
    )
    assert ids_tree == ids_direct, (
        f"tree backend wrote {len(ids_tree)} snapshots, direct wrote "
        f"{len(ids_direct)}; ids differ"
    )
    assert len(ids_tree) == snapshots

    # Snapshot times agree between backends.
    np.testing.assert_allclose(t_tree, t_direct, rtol=0.0, atol=1e-9)

    # Last snapshot is at time_end (this is what snap_every got wrong).
    expected_last = time_start + n_steps * dt
    assert t_tree[-1] == pytest.approx(expected_last, abs=1e-9)

    # First snapshot is the initial condition.
    assert t_direct[0] == pytest.approx(
        time_start if snapshots > 1 else expected_last, abs=1e-9
    )
    assert t_tree[0] == pytest.approx(t_direct[0], abs=1e-9)


def test_snapshot_steps_schedule_is_shared_formula():
    """The schedule formula itself: exactly `snapshots` entries ending at n_steps."""
    for n_steps, snapshots in [(11797, 300), (50, 7), (60, 1), (1, 1), (100, 100)]:
        if snapshots > 1:
            steps = np.round(np.linspace(0, n_steps, snapshots)).astype(int)
        else:
            steps = np.array([n_steps], dtype=int)
        assert steps.size == snapshots
        assert steps[-1] == n_steps
        assert np.all(np.diff(steps) >= 0)
        if snapshots > 1:
            assert steps[0] == 0
