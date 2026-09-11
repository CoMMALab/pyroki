#!/usr/bin/env bash
# E10 cost fitting on the recollected teleop data.
#
#   ./iosp/scripts/run_e10_ablation.sh main     # the 25-fit / 5-held run, 3 GPUs
#   ./iosp/scripts/run_e10_ablation.sh ladder   # demo-count ablation, implicit only
#   ./iosp/scripts/run_e10_ablation.sh status   # progress of whatever is running
#   ./iosp/scripts/run_e10_ablation.sh stop     # kill everything this launched
#
# Prefix any command with DRYRUN=1 to print the commands instead of running.
#
# THE SPLIT (both commands use it; see `iosp.fit.teleop.mixed_split`)
#   fit  : 20 episodes from the 2026-09-03 session + the first 5 of the
#          2026-09-02 session            = 25
#   held : the remaining 5 of 2026-09-02 =  5
# The 2026-09-03 pool is the 23 CLEAN episodes -- four of the recorded 27 are
# quarantined in `iosp/data/demos/excluded/` because their bucket sits
# 0.70-0.75 m out, past the reach of the pinned release target, and their
# rollout diverges to 195-799 rad.  See `excluded/EXCLUDED.json`.
#
# CONFIGURATION, identical to `e10_methods_perseg_r3_ms3`:
#   --pin-ik bucket      release pinned to the bucket CENTRE, grasp to the
#                        cube: the task events are constraints, only the
#                        free-space preference is fitted.  Also what makes
#                        --n-starts 3 possible -- with theta_ik constant in u,
#                        `vmap` never batches the IK ffi_call, which has no
#                        batching rule (un-pinned multistart dies on
#                        NotImplementedError).
#   --free-space-only    event rows out of the loss; the constraints fix them.
#   --per-segment        per-phase weights, K = 28 rather than 10.
#   --n-restarts 3   inner-solver restarts (best of 3 local solves per
#                    segment, inside the forward map -- NOT an outer branch)
#   --n-starts 3     outer multistart, chunked 1 at a time (see NSTARTS)
#
# RUNTIME for `main`, scaled from ms3's own wall times by batch size (30/10):
#   implicit ~1.1 h   cmaes ~10.5 h   fd ~31 h   (3 starts, chunked)
# One method per GPU, so implicit reports first and the other two run on.
# fd is the long pole and will NOT finish overnight; it checkpoints on its own
# stream, so stopping it early costs only fd.
set -u

REPO=/home/sadmin/Work/pyroffi
ENV_NAME=pyroffi              # NOT pyroffi-tamp; iosp runs in the plain env
CONDA_SH=/home/sadmin/miniconda3/etc/profile.d/conda.sh
LOGDIR=$REPO/iosp/data/logs
RESULTS=$REPO/iosp/data/results

N_TRAIN=${N_TRAIN:-20}        # episodes from the 2026-09-03 session
N_ORIG_FIT=${N_ORIG_FIT:-5}   # episodes of the original 10 moved into the fit set
# THREE outer starts: u=0 (the measured standoff prior) plus two random draws.
# There is no second outer axis -- `--n-branches` belongs to
# `iosp.fit.multistart` (e10_teleop --mode multistart), not to this
# experiment -- so 3 starts is 3 candidates, not 3x3.
#
# What OOM'd at 3 starts was not redundancy but the batch: candidates x
# episodes x n_restarts, and the episode term went 10 -> 30 since ms3.
# SEEDCHUNK=1 keeps all three starts and evaluates them ONE AT A TIME, which
# is numerically identical (the starts never interact in the forward map) and
# costs almost no wall-clock, because one 30-episode solve already saturates
# the card -- measured, batching 2 candidates bought 1.11x and 4 bought 1.17x.
# So: multistart robustness kept, peak memory back to single-start levels.
NSTARTS=${NSTARTS:-3}
NRESTARTS=${NRESTARTS:-3}

SEEDCHUNK=${SEEDCHUNK:-1}     # starts evaluated at once; 1 is what fits in 24 GiB
COMMON="--pin-ik bucket --free-space-only --per-segment --seed-chunk $SEEDCHUNK"

mkdir -p "$LOGDIR"

# launch <name> <gpu> <method> <grid> <out-dir> [tag]
#   `tag` names the checkpoint stream (`checkpoints.<tag>.jsonl`) and defaults
#   to the method.  Two runs may share an out-dir only with DIFFERENT tags --
#   that is how the ladder's rungs land beside the n=25 point of `main`, so a
#   single `make_e10_ablation --results` reads the whole implicit curve.
launch() {
  local name=$1 gpu=$2 method=$3 grid=$4 outdir=$5 tag=${6:-$3}
  local log=$LOGDIR/e10_$name.log
  local done_f=$LOGDIR/e10_$name.done
  local pidfile=$LOGDIR/e10_$name.pid

  if [[ -f $pidfile ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "  $name already running (pid $(cat "$pidfile")); skipping"
    return
  fi

  if [[ ${DRYRUN:-0} != 0 ]]; then
    echo "  [dry] $name  GPU $gpu  -> $log"
    echo "        IOSP_JAX_CACHE_DIR=$REPO/iosp/data/jax_cache_$name  XLA_FLAGS=--xla_gpu_autotune_level=0"
    echo "        python -m iosp.experiments.e10_demo_ablation --methods $method" \
         "--n-grid $grid --loop method --n-train $N_TRAIN --n-orig-fit $N_ORIG_FIT" \
         "$COMMON --n-restarts $NRESTARTS --n-starts $NSTARTS" \
         "--rollout-at all --gram-at all --out-dir $outdir --tag $tag"
    return
  fi

  rm -f "$done_f"
  mkdir -p "$outdir"
  # `setsid` detaches into its own session and process group, so these survive
  # the terminal -- and the Claude session -- that started them.
  setsid bash -c "
    source $CONDA_SH
    conda activate $ENV_NAME
    cd $REPO
    export CUDA_VISIBLE_DEVICES=$gpu
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    # PER-WORKER JAX compile cache.  The shared one is not concurrency-safe:
    # XLA's per-fusion autotune staging files get deleted out from under a
    # sibling process, which kills it during compile with a NOT_FOUND on
    # .../xla_gpu_per_fusion_autotune_cache_dir/tmp/tmp_per_fusion_cache_*.
    # See iosp/config.py::CACHE_DIR.
    export IOSP_JAX_CACHE_DIR=$REPO/iosp/data/jax_cache_$name
    # XLA's autotuner PROFILES candidate kernels, and at 30 episodes x a
    # 39-wide CMA population that probe alone asks for 11.6 GiB on top of the
    # live buffers -- it OOMs and takes the run with it, even though the real
    # tensor is under 1 GiB.  Level 0 skips profiling and uses default
    # kernels; the fits are unchanged, only kernel selection is.
    export XLA_FLAGS="--xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"
    echo \$\$ > $pidfile
    {
      echo \"=== launch \$(date -Is)  $name  gpu=$gpu  method=$method  grid=$grid ===\"
      python -u -m iosp.experiments.e10_demo_ablation \
          --methods $method --n-grid $grid --loop method \
          --n-train $N_TRAIN --n-orig-fit $N_ORIG_FIT \
          $COMMON --n-restarts $NRESTARTS --n-starts $NSTARTS \
          --rollout-at all --gram-at all --out-dir $outdir --tag $tag
      rc=\$?
      echo \"=== exit code \$rc at \$(date -Is) ===\"
      echo \$rc > $done_f
    } >> $log 2>&1
  " < /dev/null > /dev/null 2>&1 &
  sleep 1
  echo "  $name  GPU $gpu  method=$method  grid=$grid   log: $log"
}

status() {
  echo "GPUs:"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader
  echo
  shopt -s nullglob
  local any=0
  for pidfile in "$LOGDIR"/e10_*.pid; do
    any=1
    local name; name=$(basename "$pidfile" .pid); name=${name#e10_}
    local log=$LOGDIR/e10_$name.log
    local done_f=$LOGDIR/e10_$name.done
    local state="unknown"
    if [[ -f $done_f ]]; then state="finished (exit $(cat "$done_f"))"
    elif kill -0 "$(cat "$pidfile")" 2>/dev/null; then state="running (pid $(cat "$pidfile"))"
    else state="DEAD (no process; check the log)"; fi
    printf '%-18s %s\n' "$name" "$state"
    [[ -f $log ]] && grep -E '^  -> n=|^# \[|^  fit:|^  compile:' "$log" \
        | tail -2 | sed 's/^/                   /'
  done
  [[ $any == 0 ]] && echo "nothing launched yet"
  echo
  echo "checkpoints:"
  for f in "$RESULTS"/e10_*/checkpoints*.jsonl; do
    printf '  %-52s %s records\n' "${f#$RESULTS/}" "$(wc -l < "$f")"
  done
}

stop() {
  shopt -s nullglob
  for pidfile in "$LOGDIR"/e10_*.pid; do
    local name; name=$(basename "$pidfile" .pid); name=${name#e10_}
    if kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      # Negative pid kills the whole setsid process group, so the python child
      # goes too rather than being reparented and left holding the GPU.
      kill -TERM -"$(cat "$pidfile")" 2>/dev/null || kill -TERM "$(cat "$pidfile")"
      echo "  stopped $name (pid $(cat "$pidfile"))"
    fi
  done
}

case "${1:-main}" in
  main)
    OUT=$RESULTS/e10_fit25
    echo "25 fit ($N_TRAIN from 09-03 + $N_ORIG_FIT from 09-02) / 5 held"
    echo "config: $COMMON --n-restarts $NRESTARTS --n-starts $NSTARTS"
    echo "est:    implicit ~1.1 h | cmaes ~10.5 h | fd ~31 h"
    echo "out:    $OUT"
    launch fit25_implicit 1 implicit 25 "$OUT"
    launch fit25_cmaes    2 cmaes    25 "$OUT"
    launch fit25_fd       3 fd       25 "$OUT"
    echo
    echo "watch:  tail -f $LOGDIR/e10_fit25_implicit.log"
    echo "table:  conda run -n $ENV_NAME python -m iosp.analysis.make_e10_ablation --results $OUT"
    ;;
  # Tomorrow's run: the demo-count ladder, implicit only, so it costs ~1/10th
  # of doing it for all three.  Same split and config, walked DESCENDING so a
  # run stopped early keeps the large-n anchors.
  # The demo-count ladder, implicit only -- ~1/10th the cost of doing it for
  # all three methods.  Grid defaults to 20,15,10,5: n=25 is deliberately
  # ABSENT because `main` already fits it, and both write into the same
  # directory under different tags, so the five rungs aggregate as one curve.
  # Walked descending, so a run stopped early keeps the larger anchors.
  ladder)
    OUT=$RESULTS/e10_fit25
    GRID=${GRID:-20,15,10,5}
    echo "ladder (implicit only): grid $GRID   gpu ${2:-0}   out: $OUT"
    echo "est: ~2.5 h total (0.9 + 0.7 + 0.5 + 0.4)"
    echo "n=25 comes from the 'main' run's implicit stream in the same dir"
    launch ladder_implicit "${2:-0}" implicit "$GRID" "$OUT" implicit_ladder
    ;;
  status) status ;;
  stop)   stop ;;
  *) echo "usage: $0 [main|ladder [gpu]|status|stop]" >&2; exit 2 ;;
esac
