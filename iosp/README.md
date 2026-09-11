# IOSP — inverse optimal control on a composed pick-and-place task

`ioc` inverts **one** trajectory-optimization segment. `iosp` inverts a whole
**task skeleton**: IK → approach → grasp → transport → place, chained through
literal (undetached) data flow, so a single rollout of a complete pick-and-place
can be inverted for per-segment cost weights. All differentiation machinery —
implicit adjoint, finite differences, CMA-ES, subspace refit — is reused
unmodified from `ioc`.

The demonstrations are **synthetic**: every one is a rollout of this same model
at `config.THETA_IK_STAR` / `config.Z_TRAJOPT_STAR`. That is deliberate — it
makes "did recovery work?" a question with an exact answer instead of a
judgement call.

The exception is **E10**, which fits recorded human teleop instead, on an FR3
rather than the Panda. It has its own constraints, gotchas and claim structure —
see [`E10_TELEOP.md`](E10_TELEOP.md) before working on it.

## Layout

```
iosp/
  config.py         every path, task constant, ground-truth weight and solver default
  model/
    pickplace.py    the composed IK -> 4-segment trajopt forward model
    scenes.py       scene A (fit), scene B (held out), multi-scene contexts
  fit/
    params.py       the u -> theta parameterization and its softmax gauge
    parametric.py   build_parametric: the path-A bilevel forward map
    procedure.py    the 5-stage fit-wide/Gram/select/refit/report driver + G0 gate
    multistart.py   many candidates (IK branch x cost start) as one batched program
  experiments/      one module per experiment; see the table below
  checks/           diagnostics that gate the experiments
  record/           run a fit, save every outer step to .npz
  viz/              renderers that read those .npz files
  analysis/         aggregate multi-run sweeps into tables
  shelved/          deliberately out of scope, kept for the negative results
  scripts/          multi-GPU sweep drivers
  tests/            regression tests (pytest); run `pytest iosp/tests/`
```

Nothing in `config.py` imports from `iosp`. Before this layout, `study3`
imported its URDF paths from `study0_segment_ablation`, its ground truth from
`recovery_bench` and its held-out offsets from `generalization_check` — three
experiment scripts had to be importable for a fourth to run.

## The core idea, in one paragraph

The inner problem is `x*(θ,c) = argmin_x Σ_k θ_k φ_k(x,c)`; the outer problem
fits θ so that `x*` matches the demonstration. Differentiating through the
inner **solution** needs `dx*/dθ`, which exists only where `x*(θ)` is smooth.
Two things break that smoothness on a redundant arm, and both are handled
structurally rather than by tuning: the redundant IK's winner selection (fixed
by `IK_CONTINUITY_WEIGHT`, and by pinning one branch per candidate) and the
trajopt's non-convexity (handled by running many candidates and selecting once,
at the end, on training loss). Everything else — rank deficiency, the softmax
gauge, the identifiable-subspace refit — is about *which directions of θ the
demonstration can resolve at all*, and lives in `fit/procedure.py`.

## Experiments

Run each as `python -m iosp.experiments.<name>`. All GPU work needs
`CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false` — check
`nvidia-smi` first, these boxes are shared. The persistent compile cache
(`config.enable_compilation_cache()`) turns a ~25 min cold compile into ~2 min.

| module | old name | question | headline finding |
|---|---|---|---|
| `recovery_bench` | — | Can the implicit adjoint recover a known θ\*, and how does it compare to CMA-ES at equal solve count? | The reference benchmark. FD agreement is *not* the standard here; recovering a known θ\* is. |
| `e0_segment_ablation` | study0 | Freeze all-but-a-growing-subset of the 4 segments at ground truth — where in the chain does recovery break? | Composition is **not** the problem. Fitting only `transport` errs as much as fitting all four (~0.21–0.24). |
| `e0d_eigen_projection` | study0d | Split recovery error into its identifiable and null components. | The optimizer is behaving correctly: top-1 error 0.0202, null (8-dim) 0.2422. Raw ‖θ̂−θ\*‖ cannot tell these apart. |
| `e1_minimal_identifiable` | study1 | A deliberately identifiability-clean K=3 transport-only problem. | The minimal case where recovery *should* work; the loss floor is a basin, not a floor (multistart reaches 0.13128 vs 0.44543). |
| `e1b_multidemo` | study1_diagnostic_multidemo | Does adding demonstrations fix it? | Only mildly and non-monotonically — more demos help only if they change the *span* of the Gram. |
| `e1c_fd_check` | study1_diagnostic_fd_check | FD vs implicit adjoint on E1's K=3 loss. | Diagnostic only. See `checks/composed_fd.py` for the composed-chain version. |
| `e2_demo_quality` | study2 | Does recovery track demo **curation** rather than demo **count**? | Curation. Count barely moves it. |
| `e3_identifiable_refit` | study3 | Fit wide → Gram → select r → refit on `U_r`. Does rank deficiency become a *generalization* cost instead of a *reconstruction* cost? | The main path-A experiment. Refit pins the null component by construction; report in the `U_r` projection, never as raw L2. |
| `e4_three_stage` | study4 | Invert spasm's three-stage forward pass (segments + a whole-trajectory refine). | Not yet validated — do not use it for a figure. |
| `e5_tamp2d` | study5 | The same tied three-stage inversion on a cheap 2D TAMP benchmark. | The drawable sanity check for the composed method. |
| `e7_loss_space` | scratch/joint_loss_test (now e7) | Score the outer loss in **joint** space instead of EE space. | Reconstruction −18.5% on 5/5 seeds; **generalization not established** (−3.5% ± 21.9%, 4/5). Mechanism is reconditioning (λ₁/λ₂ 30 → 4.05), *not* added rank. |
| `e8_tetris` | — | Tetris-style packing variant of the composed task. | Stress test for multi-object composition. |
| `e9_tower` | — | Tower-stacking variant of the composed task. | Vertical composition stress test. |
| `e10_teleop` | — | Recovery from **human** teleop demos (FR3 + GELLO). | Behavioural claim only — no ground-truth θ*. See `E10_TELEOP.md`. |

Numbered names are kept as stable IDs because the logs, notes and memory all
refer to them; the old `study<n>` filenames map to the table above.

## Checks — run these when a result looks wrong

| module | asks |
|---|---|
| `checks/identifiability` | What is the Gram spectrum on this scene? Which features are resolvable at all? |
| `checks/generalization` | Does fit-demo RMSE actually indicate correct recovery, or only memorization? |
| `checks/fullchain` | Does a loss still differentiate through IK → 4 chained solves? |
| `checks/ik_branch` | Are the behavioural-loss spikes IK self-motion branch flips? (They were.) |
| `checks/branch_classes` | How many *distinct* IK branches does this arm actually have here? |
| `checks/composed_fd` | Do the single-segment soft-flag findings carry to the composed chain? |
| `checks/loss_rmse_consistency` | Is the rollout a stable function of `u`? (The G0 gate — catches disagreements stationarity screening misses.) |
| `checks/forward_extract` | Extract joint paths at `u=0` for all three domains → `.npz`. |
| `checks/dynamic_report` | Drive those `.npz` paths through MuJoCo physics and score task success. |
| `checks/pickplace_sim` | Standalone FR3 pickplace build + physics test (no `.npz` intermediate). |

## Reproducing the current results

**The multistart robustness result** (the strongest claim: 9 candidates,
held-out scene at 2× displacement, one selection on training loss):

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m iosp.record.multistart --space joint --n-branches 3 --n-starts 3 \
      --steps 40 --scene-b-scale 2.0 --out iosp/data/viz/multistart_behavior.npz
```

~2 h. Prints the per-candidate table and writes an `.npz` holding every
candidate's path at every outer step. Then render:

```bash
python -m iosp.viz.multistart iosp/data/viz/multistart_behavior.npz   # talk figure
python -m iosp.viz.behavior3d iosp/data/viz/multistart_behavior.npz   # 3D, single fit
python -m iosp.viz.behavior   iosp/data/viz/multistart_behavior.npz   # 2D, x-y projection
```

**The joint-vs-EE loss comparison**, paired by seed:

```bash
python -m iosp.experiments.e7_loss_space --space joint --seed 0
python -m iosp.experiments.e7_loss_space --space ee    --seed 0
```

~25 min each warm (add ~11 min for the sensitivity spectrum unless you pass
`--no-spectrum`).

**The whole multi-seed sweep** across every free GPU, then the tables:

```bash
bash iosp/scripts/queue_multiseed.sh          # ~3.5 h on 4 idle GPUs
python -m iosp.analysis.multiseed             # prints stage A and stage B tables
```

`IOSP_RESULTS=<dir>` repoints the aggregator if you keep results elsewhere.

## Human teleoperation reconstruction

`e10_method_comparison` fits the composed planner to **human** GELLO teleop
episodes and compares the outer-loop methods on it. There is no `theta_star`
here, so every parameter-recovery metric is undefined and the claim is purely
behavioural — read `E10_TELEOP.md` before changing anything in the pipeline.
Run all of this in the `pyroffi` env (`pyroffi-tamp` if you need the CUDA FFI
kernels; `spasm-pyroffi`'s JAX 0.8.0 cannot load them).

### 1. Collect the demonstrations

Recording lives in the sibling repo, not here:

```bash
cd ../sim_teleop
python record.py --port /dev/ttyUSB0            # the real GELLO leader
python record.py --mock                         # no hardware, scripted sweep
```

Browser viewer on `--viser-port` (default 8080) so it works over SSH. Three
buttons: start, stop-and-save, stop-and-DISCARD. **Use discard freely** — about
half of hand-teleoperated attempts fumble the grasp or knock the cube off the
table, and a failed episode is not a noisy sample of the intended behaviour, it
is a sample of *different* behaviour. Nothing hits disk until stop-and-save.

Episodes land in `../sim_teleop/data/demos/ep_<timestamp>/`, each with
`state.jsonl` (what the fit reads), `episode.npz` (velocities/torques, recorded
but unread) and `factors.json` (the randomisation record — an episode without
it can be replayed but not fitted). `sim_teleop/pickplace/iosp_export.py`
collapses each to an `(N_FULL, 7)` waypoint path; it imports `N_FULL`/
`PHASE_SPAN` from this package, so do not vendor a copy here.

The current ten episodes are `ep_20260902_09*`, split 8 fit / 2 held-out
chronologically.

### 2. Fit

The configuration behind the current per-segment result:

```bash
CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  python -m iosp.experiments.e10_method_comparison \
      --pin-ik bucket --free-space-only --per-segment \
      --n-restarts 3 --n-starts 3 \
      --methods implicit,fd,cmaes --compile-timeout 120 \
      --out-dir iosp/data/results/e10_methods_perseg_r3_ms3
```

- `--per-segment` gives each of the 4 segments its own 6 weights: K=28 = 4
  pinned `theta_ik` + 4x6. Without it, K=10 (one shared set of 6).
- `--pin-ik bucket` makes the release/grasp events constraints instead of fitted
  weights; `measured` fits better but releases on the rim (0/10).
- `--n-restarts` is the *inner* solver restarts per segment; `--n-starts` is the
  *outer* multistart, selected on TRAINING loss only.
- `--methods` — drop `unrolled` unless you raise `--compile-timeout`; it hits
  `compile_timeout` at 120 s and is written out as `diverged` with no theta.
- Runtimes on the current data: implicit ~21 min, CMA-ES ~3.5 h, FD ~10 h.

Writes `summary.json`, `joint_paths.npz`, `paths.npz`, `u_hats.npz`.
`--out-dir` defaults to `iosp/data/results/e10_methods`, which overwrites — pass
it explicitly for anything worth keeping.

### 3. Visualize

Side-by-side physics playback of every method's reconstruction in one MuJoCo
scene, replayed from the saved `joint_paths.npz` (nothing is refitted):

```bash
MUJOCO_GL=egl PYTHONPATH=. python -m iosp.viz.e10_methods_viser \
    --results-dir iosp/data/results/e10_methods_perseg_r3_ms3 \
    --methods demo,fd,cmaes,implicit \
    --episode-index 9 --port 8081
```

- **Set `--methods` to what the run actually contains.** The default order
  includes `unrolled` and `init`; a missing method is dropped with a warning,
  but `init` only resolves if some method's `u_hat` is exactly 0. When none is
  (the usual case with `--n-starts > 1`), pass `--recompute-init` to roll the
  baseline out here, which rebuilds the forward map and needs a GPU.
- `--episode-index 0-9`; indices `>= summary["n_fit"]` (8, 9) are the held-out
  demos, which are the interesting ones once rollout success saturates.
- Pick a free `--port`; 8080 is often taken by another viser.
- Robot and cube share a hue per method, so a failure reads as "the GREEN arm
  dropped the GREEN cube on the table".

### 4. Render the table

```bash
python -m iosp.analysis.make_e10_table \
    --summary iosp/data/results/e10_methods_perseg_r3_ms3/summary.json \
    --out iosp/figures/e10_perseg_r3_ms3.tex
```

Emits the joint/EE RMSE x fit/held-out block plus task success and wall clock;
it reads K from the summary, so it is basis-agnostic (verified on both K=10 and
K=28). Task success is the cube-in-bucket count under contact physics — a
VERIFICATION metric, never part of the fitting loss. Note it saturates at 8/8,
2/2 including the `u=0` baseline on the current data, so it does not
discriminate methods; report the joint-space RMSE as the headline.

## Forward-solve verification (dynamic feasibility)

After fitting (or at the `u=0` baseline), verify that the planned joint paths
actually execute under contact physics — the arm picks the object up and places
it where the skeleton says. This catches issues that kinematic replay hides:
servo tracking lag, joint-limit clipping, collisions with stacked geometry, and
release targets that land outside the bucket.

All three domains share the same two-stage pipeline:

### 1. Extract joint paths

```bash
CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m iosp.checks.forward_extract <domain> --out scratch/feas/<domain>.npz
```

where `<domain>` is `tetris`, `tower`, or `pickplace`. Each calls its own
experiment's `build()` and rolls out `paths_fn(u=0)` to produce an `.npz` with
`q` (joint paths), pick/place targets, and domain metadata.

- **tetris**: `e8_tetris.build`, Panda, 3-block packing, 60 L-BFGS iters.
- **tower**: `e9_tower.build`, Panda, 6 synthetic scenes, 60 iters.
- **pickplace**: `build_pick_and_place` (FR3 teleop scenes), `pin_ik="bucket"`,
  `freeze_ik=True`, 600 iters. This is the same pipeline `e10_method_comparison`
  uses — the robot is an FR3 and the scenes are the recorded teleop episodes.

### 2. Run through MuJoCo physics

```bash
MUJOCO_GL=egl python -m iosp.checks.dynamic_report scratch/feas/*.npz
```

Reads each `.npz`, builds the domain's MuJoCo scene (pedestal + carried object
for pickplace, goal region for tetris, stacked blocks for tower), drives the
arm through the waypoints under gravity, and reports per-scene success:

- **tetris** / **tower**: scored on the object's *settled* position after the
  arm releases and the block comes to rest. Success = within 5 cm of target.
- **pickplace (FR3)**: scored on bucket landing via `e10_spasm_sim.physics_success`
  — the cube must end inside the bucket under contact-driven physics (no
  kinematic attach hack). Reports `dxy` (horizontal distance from bucket centre)
  and `dz` (height above bucket floor).

The report also breaks tracking error into joint-limit overshoot vs physical
obstruction, which tells you whether a failure is the plan's fault or the
scene's.

### Current baselines (u=0)

| domain    | success | notes |
|-----------|---------|-------|
| tetris    | 5/6     | one scene has a tight packing that clips a wall |
| tower     | 4/6     | arm collides with already-stacked blocks during dynamic execution |
| pickplace | 10/10   | with `pin_ik="bucket"` and 600 iters |

### 3. Visualize the rollout

Each domain has a viser-based 3D viewer that replays the `.npz` in a MuJoCo
scene with contact rendering, a playback scrubber, and per-row contact reports.
No GPU needed — the viewer reads the saved joint paths.

```bash
# tetris — shows goal region, packed tetrominos, contact depths
python -m iosp.viz.tetris_viser --from-npz scratch/feas/tetris.npz

# tower — shows stacking base, existing tower, obstacle spheres
python -m iosp.viz.tower_viser --from-npz scratch/feas/tower.npz

# pickplace (Panda synthetic) — shows pedestal, carried box, obstacle
python -m iosp.viz.pickplace_viser --from-npz scratch/feas/pickplace.npz
```

Opens a browser viewer (default port 8080; pass `--port <N>` if taken). Multiple
scenes are laid out side-by-side. The playback panel reports the deepest contact
on each row — the `feasibility_report` verdict, frame by frame.

For FR3 teleop pickplace specifically, `e10_methods_viser` shows side-by-side
physics playback of multiple methods' reconstructions in the teleop scene:

```bash
MUJOCO_GL=egl python -m iosp.viz.e10_methods_viser \
    --results-dir iosp/data/results/e10_methods_perseg_r3_ms3 \
    --methods demo,fd,cmaes,implicit \
    --episode-index 0 --port 8081
```

<!-- ### Standalone pickplace check

`pickplace_sim.py` is a lighter-weight script that builds and tests pickplace
in one step (no `.npz` intermediate), useful for quick iteration:

```bash
CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  python -m iosp.checks.pickplace_sim [--n-iters 600] [--pin-ik bucket]
```

Pass `--u '<json list>'` to test with recovered weights instead of `u=0`. -->

## Reading the results honestly

Three rules, each of which was learned by getting it wrong first:

1. **Score behaviour, not parameters.** The spectrum is rank-deficient, so the
   fit reproduces behaviour while drifting along null directions. A weight-bar
   plot against ground truth looks like failure on a good fit.
2. **Report at a fixed step budget, never at each run's own minimum.** Taking
   the min is selection on the held-out criterion.
3. **The eigendecomposition is local to a basin.** `G = JᵀJ` needs `x*(θ)`
   differentiable, so it is blind to basin mismatch — a wrong-branch candidate
   can have a perfectly conditioned Gram and still sit 1 m from the demo. Rank
   diagnoses null-drift and identifiable error; only multistart diagnoses basin
   mismatch.

## Theory

`THEORY.md` (the bilevel problem and the three ways to get `dx*/dθ`),
`THEORY_IDENTIFIABLE_REFIT.md` (the rank-deficient refit derivation, path A and
path B), `THEORY_NONPARAMETRIC_GAPS.md`. `HANDOFF.md` is the running
investigation log.
