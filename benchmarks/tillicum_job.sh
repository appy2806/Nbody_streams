#!/bin/bash
#SBATCH --job-name=nbs_bench
#SBATCH --qos=debug
#SBATCH --gpu-bind=closest
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
# Benchmarks on one exclusive H200 (Tillicum).  Submit from the repository copy
# in scratch (REPO), one task per job:
#   sbatch --export=ALL,REPO=$PWD,TASK=sweeps benchmarks/tillicum_job.sh
#   sbatch --export=ALL,REPO=$PWD,TASK=maxn_call,NS="1e8 2e8 3e8 4e8 5e8" benchmarks/tillicum_job.sh
#   sbatch --export=ALL,REPO=$PWD,TASK=maxn_run,NS="1e8 2e8 2.7e8" benchmarks/tillicum_job.sh
#   (add MEMFRAC=0.95 to the maxn tasks to call nbody_streams.tree_gpu.set_memory_fraction)
#   sbatch --export=ALL,REPO=$PWD,TASK=validate benchmarks/tillicum_job.sh   # N > 2^28 check
# PY: python with cupy (default: the Nbodystream conda env).
set -u
REPO=${REPO:?set REPO to the repository copy}
PY=${PY:-/gpfs/home/arora125/.conda/envs/Nbodystream/bin/python}
module load cuda/13.0.0 > /dev/null 2>&1
C=$REPO/.bench_cache
export CUPY_CACHE_DIR=$C/cupy NUMBA_CACHE_DIR=$C/numba MPLCONFIGDIR=$C/mpl XDG_CACHE_HOME=$C
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$REPO TMPDIR=${TMPDIR:-$C/tmp}
mkdir -p $C/tmp
echo "== $(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) $(date)"
(cd $REPO/nbody_streams/tree_gpu && { [ -f libtreeGPU.so ] || make -j8 2>&1 | grep -E "Primary arch| error"; })
$PY -c "import nbody_streams; print(nbody_streams.__file__)"
cd $REPO
case ${TASK:?set TASK} in
  sweeps)
    $PY -u benchmarks/timing_sweep.py --method gpu_tree --label h200
    $PY -u benchmarks/timing_sweep.py --method gpu_direct --label h200 --max-call 10 ;;
  maxn_call)
    $PY -u benchmarks/max_n.py --mode call --label h200 --n ${NS:?} ${MEMFRAC:+--mem-fraction $MEMFRAC} ;;
  maxn_run)
    $PY -u benchmarks/max_n.py --mode run --label h200 --n ${NS:?} ${MEMFRAC:+--mem-fraction $MEMFRAC} ;;
  validate)
    $PY -u benchmarks/large_n_check.py ${N1:-4e8} ${N2:-2.7e8} $C/tmp ${MEMFRAC:-} ;;
  *) echo "unknown TASK=$TASK"; exit 1 ;;
esac
echo "== DONE $(date)"
