# Benchmarks

Scripts behind the Performance section of the top-level README.

| Script | What it measures | Output |
|---|---|---|
| `timing_sweep.py` | Time per force evaluation against N for CPU direct (Numba), FMM (pyfalcon), GPU direct and GPU tree; Plummer sphere, softening 0.01 a, theta 0.6 | `data/sweep_<label>_<method>.json` |
| `max_n.py` | GPU tree at large N: time per call (`--mode call`) or per `run_nbody_gpu_tree` step (`--mode run`) and peak GPU memory, one subprocess per N | `data/maxn_<label>_<mode>.json` |
| `plot_timings.py` | The README figure `plots/acceleration_timings.png` from the sweeps | PNG |
| `large_n_check.py` | Correctness above 2^28 particles: tree forces of high-index particles against a float64 direct sum (H200) | stdout |
| `tillicum_job.sh` | Slurm job for the H200 runs (`TASK=sweeps`, `maxn_call`, `maxn_run`, `validate`) | |

Labels used in the figure: `rtx3080` (RTX 3080 Laptop + i9-11980HK, also the
CPU and FMM lines), `l40`, `h200`.

```bash
python benchmarks/timing_sweep.py --method gpu_tree   --label rtx3080
python benchmarks/timing_sweep.py --method gpu_direct --label rtx3080 --max-call 10
python benchmarks/timing_sweep.py --method cpu_direct --label rtx3080 --max-call 10
python benchmarks/timing_sweep.py --method fmm        --label rtx3080
python benchmarks/max_n.py --mode call --label rtx3080 --n 1e7 2e7 3e7
python benchmarks/plot_timings.py
```

Each point keeps the minimum and median call time; the figure uses the
minimum, which is the most robust number on a GPU shared with other work.
Run scripts from the repository root (or put it on `PYTHONPATH`) so the
checked-out code, not an installed copy, is measured.
