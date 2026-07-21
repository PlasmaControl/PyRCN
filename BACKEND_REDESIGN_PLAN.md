# PyRCN Backend Redesign — Living Plan

> Status: **Backend redesign shipped and released.** Torch backend behind
> ESN*/ELM* (fast path + NumPy fallback), public `pyrcn.nn`, gradient and
> fully-trainable training (readout / reservoir / input). Merged `dev → main`
> and **released as `0.1.0` on PyPI**. Documentation fully modernized on
> `origin/dev`. **Active task: raise test coverage toward 100% + a read-only
> performance-bottleneck analysis** (see "Current work"). Edited incrementally.

## Current work (active task — resume here after compaction)

**Requested 2026-07 (post-release). Constraints (user emphatic):** any
optimization must be **numerically identical within ~1e-12 on float64**
(mathematically-equivalent float reordering OK, nothing looser; verify by
capturing current output on fixed seeds and asserting equality after); **no
performance regression**; full suite stays green; heavy reading delegated to
**subagents** to avoid context compaction.

### Goal A — coverage → ~100%. DONE.
**86.2% → 99.60%** (commit `d2b2f79`, local, not pushed). Additive tests only;
the only src edits are justified `# pragma: no cover` markers. Full suite green,
flake8 clean. Residual: 2 lines blocked by the NormalDistribution.fit bug + 9
hard partial branches. Reports at `<scratchpad>/cov_missing.txt`/`cov.json`
(before) and `cov_after.*`.

### Goal B — performance. Analysis DONE (4 read-only subagents). Implementing per approval.

**Method (TDD) + strict acceptance gate for EVERY point below:** write the
parity test first, then a before/after benchmark. Keep the implementation ONLY
if the parity test passes (identical within ~1e-12, bit-exact where possible)
AND a speed-up is actually measured. If EITHER fails (no measured speed-up, or
parity fails), **discard/revert the change and leave that point OPEN here**
(mark status). One change at a time, committed separately, full suite green.
See memory `perf-work-tdd-parity`.

Ranked in-scope findings (all numerically identical within ~1e-12 unless noted):
- **P1 — DONE (commit `a72ad3a`, local): two-tier reservoir dispatch**
  (`nn/_reservoir.py`). Fast sub-case (`leakage==1.0` & tanh/relu, fixed
  weights) → ATen fused `torch.rnn_tanh`/`rnn_relu`. General case (leaky /
  Euler / logistic / identity / bounded_relu / trainable) →
  `torch.jit.script` recurrence. Parity: `tests/test_reservoir_parity.py`
  (200-config grid vs captured fixture) — general bit-exact, fused ≤1e-12, max
  diff 0.0. Independent before→after benchmark: fused ~3.9×, general ~2.8×;
  full suite exit 0, flake8+mypy clean. (Note: invalid-config fits in a param
  search now raise a different RuntimeError message from the fused op, but
  still raise → same fits fail with same nan scores; no numerical change.)
- **P2** strip `nn.Module`/`RNNCell` per-step dispatch — subsumed into P1's
  scripted general path.
- **P3** numpy readout `inv(K+αI)@xTy` → `np.linalg.solve`
  (`linear_model/_incremental_regression.py`) — 2-3.5×, diff ≤1.7e-16.
- **P4** torch readout LU `solve` → Cholesky (`nn/_readout.py`) — ~1.4×, diff
  ≤1.25e-16, guard `alpha≤0` → fallback to solve.
- **P5** drop the wasted numpy `input_to_node.transform` fed to
  `node_to_node.fit` (only `shape[1]` used) in `_esn.py`/`_elm.py` — ~10% of
  single-series fit; `np.array_equal` verified.
- **P6** hoist `EulerNodeToNode`'s per-step `scaling·W + γ·I` rebuild
  (`base/blocks/_node_to_node.py`) — ~9× on the Euler path, bit-identical.
- **P7** drop `InputToNode`'s `np.ones` bias temp
  (`base/blocks/_input_to_node.py`) — bit-identical.
- **P8** `np.dot` instead of `safe_sparse_dot` for dense (`_node_to_node.py`) —
  ~4%, bit-identical.
- **P9** (medium risk) de-dup `concatenate_sequences` conversions
  (`util/_util.py`) — needs before/after array-equality gating.
- **P10** (larger) batched reservoir with `lengths`-masking — avoids ~37%
  padding waste; only relevant if the batched `check_sequences` path is wired in
  (estimators currently run per-sequence, batch=1).
- **Thread oversubscription (biggest real-world effect, NOT bundled as
  identical):** default multi-thread BLAS makes the per-step reservoir loop
  pathologically slow (a single 8000-step series did not finish in 90 s at 48
  threads — the cause of prior runaway CPU). Capping intra-op threads gives ~3×
  on ragged workloads, BUT changing thread count reorders matmul reductions →
  may exceed 1e-12 on long chaotic recurrences. Treat as an env/deployment knob
  or verify per case. P1's fused kernel sidesteps most of it.
- **Out of scope (change results > 1e-12):** `spectral_normalize`
  eig→iterative; `fan_in` RNG-loop vectorization; folding `spectral_radius`
  into `W` (rel ~1.8e-12, and the numpy path is the parity oracle); float32 /
  GPU cuDNN.
- Order after P1: P3, P5, P6, P7 are the next easy bit-identical wins; P4/P8
  low-med; P9/P10 deferred.

### Bugs found during the coverage push (PRE-EXISTING; NOT fixed — a fix changes behavior, needs separate go-ahead)
- **`postprocessing/_normal_distribution.py`:** `fit` calls
  `scipy.stats.norm.fit(X=…, y=…)`; those kwargs do not exist → **always raises
  TypeError**, estimator unusable. (Blocks 2 coverage lines.)
- **`metrics/_classification.py`:** `log_loss` forwards `eps=` (removed in
  sklearn ≥1.5) → TypeError; `brier_score_loss` passes `y_prob=` (renamed to
  `y_proba`) → TypeError. Both wrappers broken; coverage tests currently assert
  the TypeError.
- **`preprocessing/_coates.py`:** `inverse_transform` passes a 3-D array to
  StandardScaler/PCA (require ≤2-D) → breaks when `normalize=True`/`whiten=True`;
  and `_inverse_preprocessing` applies un-normalize/un-whiten in the wrong order.
- **`linear_model/_incremental_regression.py:199`:** `normalize=True` via
  `partial_fit` is a no-op (the scaler-transformed result is discarded, not
  reassigned).
- **`base/blocks/_input_to_node.py:~494`:** `if bound_low == np.inf` is almost
  certainly meant to be `-np.inf` (dead code as written; pragma'd).
- **`util/_util.py`:** `concatenate_sequences` raises on a ragged plain `list`
  under NumPy 2 (`np.asarray` inhomogeneous shape); only object-arrays /
  equal-length lists work.
- **`model_selection/_search.py`:** `SHGOSearchCV.fit` iterates the raw
  `self.cv` attribute instead of the checked splitter → crashes on `cv=None` /
  `cv=int`.

## Status summary

**Completed**
- Phase 1 modernization (branch `modernize-sklearn`; PR #62 to `dev`).
- Phase A (A0-A6) + A4b: torch backend behind ESN*/ELM* (fast path + NumPy
  fallback for arbitrary sub-estimators); `washout`;
  `predict(initial_state=, return_state=)`.
- P1: public, documented `pyrcn.nn`; "implementation detail" disclaimer
  removed.
- Phase B: B1 (gradient readout solver + `random_state` reproducibility +
  tight closed-form consistency demo), B2 (trainable reservoir, BPTT),
  B2+ (`trainable_input`, BPTT mini-batching over sequences). Optimizers
  {adam, adamw, sgd, rmsprop, adagrad}, losses {mse, mae, huber}.
- Examples updated to the current API (no torch usage introduced).
- Activation set unified to the five supported by both backends (dropped the
  NumPy-only `softmax`/`softplus`).
- GitHub issues addressed locally: #54 (activation validation), #61
  (estimators run; notebooks fixed).
- **Release: `backend-redesign` squash-merged to `dev`; `dev` fast-forwarded to
  `main`; version bumped to `0.1.0` and published to PyPI** via a GitHub Release
  triggering `python-publish-pypi.yml` (after fixing the `PYPI_API_TOKEN`
  secret; a first attempt 403'd on a malformed token). `0.1.0` is immutable.
- **Documentation fully modernized (on `origin/dev`, tip `5c22002`; not yet
  merged to `main`):** polished RTD theme (brand accent, logo, SVG favicon,
  `sphinx-design` cards, intersphinx, MathJax); rebuilt landing page (badges
  incl. CI-generated coverage badge, capability cards, DOE + ESF funding);
  installation (Python 3.10+, extras, torch CPU/GPU); RC introduction with a
  generated ESN schematic; getting-started (verified doctests + gradient
  snippet + regenerated plots); tutorials; development/contributing;
  **API reference converted to full autosummary** (per-object stub pages,
  gitignored). Citation updated to the EAAI 2022 article everywhere. Copyright
  unified: **TU Dresden 2020-2024; Plasma Control Group, Princeton University
  2024-2026** (LICENSE, docs two-line footer, README license section). All
  builds 0 warnings; 24 doctests pass.

**Open / deferred**
- B3 demonstrations: gradient/trainable example notebooks. Not started.
- Issue #53 (multiple reservoirs, one readout): under discussion.
- Docs are on `dev`; **no `dev → main` PR opened yet** (user declined for now).
- GitHub issue comments (#54, #61) held until merge.
- Audio examples (`f0_extraction`, `multipitch_tracking`) remain
  non-functional: dataset loaders (`fetch_ptdb_tug_dataset`,
  `fetch_maps_piano_dataset`) intentionally not restored.

## Goal
Keep the **frontend** (the scikit-learn-compatible API) unchanged; replace the
**backend** so that sequential data is a first-class concern rather than a set
of workarounds bolted onto scikit-learn's stateless, 2-D-array contract.

## Hard constraints
- Public API preserved: `ESNRegressor` / `ESNClassifier` / `ELMRegressor` /
  `ELMClassifier` with `fit` / `predict` / `predict_proba` / `get_params` /
  `set_params`; `InputToNode` / `NodeToNode` blocks as configuration;
  `IncrementalRegression`; `SequentialSearchCV`; `metrics`; `datasets`.
- Runs on the modern stack (scikit-learn 1.9 / NumPy 2 / SciPy / Python 3.10+).
- **INV-1 (2-D behavior — structural, not bit-exact):** a 2-D `(T, F)` input
  keeps today's *semantics* — one continuous reservoir pass (no mid-stream
  reset) and `predict` returns a plain `(T, n_targets)` ndarray, **not** a
  length-1 object array. Numeric outputs need only match the historical
  float64 result **within a tolerance** (relaxed per D3; float32 drift allowed).

## Current-state audit (summary; full detail in memory `pyrcn-architecture`)
- Sequences smuggled as 1-D `object` arrays; sequence branch bypasses
  `validate_data`. Detection via the fragile `X.ndim == 1`; equal-length 3-D
  input is rejected.
- Reservoir is a pure-Python per-timestep loop; batching is Python-level
  (per-sequence AND per-timestep); parallelism only via heavyweight joblib-loky.
- Transformers are stateless: every `transform` resets state to zero → no
  washout, no initial-state carry, no streaming inference; the start transient
  pollutes the readout fit.
- `predict` returns object arrays → the whole custom `metrics/` module exists to
  cope. `ESN/ELM` hand-roll `get_params`/`set_params` via private
  `_get_param_names`.
- Worth preserving: incremental normal-equations readout (`K`, `xTy`, single
  solve, mergeable via `__add__`); the block-composition model.

## Design axes (to decide, incrementally)
1. **Sequence data model** — **PUBLIC CONTRACT DECIDED** (see D1 + section below);
   internal canonical form still open (coupled to axis #2).
2. **Reservoir engine** — **DECIDED: PyTorch** (see D2 + section below).
3. **State semantics** — **DECIDED (D6 washout, D7 state carry).**
4. **Params / `clone`** — **DECIDED (D8): nested params + `**kwargs` alias.**
5. **Numerics** — **DECIDED (D3): float32 allowed, INV-1 relaxed to tolerance.**

## Decisions log
- **D1 (public sequence contract): accept-all + normalize.** `fit`/`predict`
  accept several input forms and normalize internally to one canonical form.
  2-D-means-one-sequence **confirmed**, conditional on INV-1 (no change to the
  current 2-D behavior, incl. plain-array output).
- **DECIDED:** `(k,)`-target ambiguity → auto-detect (length-based) by default,
  with an explicit estimator param `task` in
  `{'auto','sequence-to-sequence','sequence-to-value'}` to force it; raise a
  clear error only on an unresolved collision (`k == Lᵢ == n_targets`).
- **D2 (reservoir engine): PyTorch.** The reservoir/`NodeToNode` (and the ELM
  feature map) run as batched PyTorch tensor ops; reservoir weights are fixed
  buffers (`requires_grad=False`), so autograd is unused but batching + optional
  CUDA are the win. Reintroduces torch as a backend dependency (reverses the
  Phase-1 removal — this is the deliberate post-upgrade `nn` plan; build clean,
  not the old draft).

## Reservoir engine (PyTorch) — from D2

### What it settles
- Internal canonical form → **padded batch `(N, L_max, F)` + `lengths`** (and/or
  `torch.nn.utils.rnn.pack_padded_sequence`), which is the natural PyTorch shape
  for batched recurrence. (Resolves the axis-1 "internal form" open item.)
- Time loop becomes a batched tensor recurrence over `L_max` (mask/packed to
  respect `lengths`), replacing the per-sample + per-sequence Python loops.

### Sub-decisions
- **D4 — torch is a HARD dependency.** Single code path, no NumPy fallback
  engine. Re-add torch to `pyproject` core `dependencies` (reverses the Phase-1
  removal). `device=` still selects cpu/cuda.
- **D3 — allow float32; INV-1 relaxed to a tolerance.** Backend may run in
  float32 (default torch dtype / GPU-friendly); historical float64 parity is not
  required bit-exact, only within tolerance. Expect a `dtype=` knob (float32/64).
  - **Verification strategy (honest despite chaotic float drift):** a reservoir
    is a dynamical system, so float32-vs-float64 states can *diverge* over long
    sequences — a raw-state tolerance is unreliable. Validate the port by
    comparing **torch-float64 vs the old NumPy-float64** backend (expect tight
    agreement) to prove correctness; treat float32 as a separate speed/precision
    mode judged on task-level metrics, not state equality.
- **Resolved:**
  - **D10 — dense `Parameter` weights.** Store input/recurrent weights as dense
    torch `Parameter`s (zeros where `k_in`/`k_rec` init made them sparse). Dense
    matmul is GPU-ideal and gradient-friendly (sparse autograd is limited);
    `predefined_*_weights` load into dense `Parameter`s. Sparse storage may be
    added later as an opt-in for very large reservoirs.
  - Readout location and device placement are resolved by **D5** below.

## Sequence data model (from D1)

### Accepted public input forms → normalization
- **2-D `(T, F)`** → **one** sequence of length `T` (continuous series).
  *Preserves* the current single-sequence/time-series behavior (e.g.
  `mackey_glass`): the reservoir runs over all rows with no mid-stream reset.
  Rows are NOT independent tabular samples for the recurrent path.
- **list / tuple / 1-D `object` array of 2-D `(Lᵢ, F)`** → **N** sequences
  (ragged). Object-array form kept for back-compat.
- **3-D `(N, L, F)`** → **N** equal-length sequences (newly allowed).
- Everything collapses to a canonical *"batch of N≥1 sequences"*.

### Targets `y` → task kind (replace the length heuristic)
- `y[i]` shaped `(Lᵢ,)` or `(Lᵢ, n_targets)` ⇒ **sequence-to-sequence**.
- `y[i]` scalar or `(n_targets,)` (one label per sequence) ⇒
  **sequence-to-value**. Determined by rank/shape, not by
  `not np.any(len_X == len_y)`.

### Implementation note
- A single validating normalizer (working name `check_sequences`) replaces
  `concatenate_sequences` + `_check_if_sequence` + `_check_if_sequence_to_value`.
  It returns the canonical batch plus per-sequence lengths and the task kind,
  and is the one place object/ragged input is handled (so estimators no longer
  bypass validation ad hoc).
- Must stay an sklearn-indexable of length `N` (for `SequentialSearchCV`/CV):
  `len` = N, `X[idx]` selects sequences.

### Still open (couple to axis #2 — reservoir engine)
- Canonical **internal** form: concatenated `(ΣLᵢ, F)` + boundaries (memory-lean,
  loop/streaming-friendly) vs. padded `(N, Lmax, F)` + mask (vectorized-batch
  friendly) vs. list-of-arrays kept as-is (bucketed batching).

## Frontend / Backend architecture (D5)

**D5 — boundary at the estimator edge (thin frontend).** NumPy↔torch conversion
happens once, at estimator `fit`/`predict` entry/exit; everything internal is
torch and stays on `device`.

**Frontend (NumPy, sklearn — the kept public API)**
- Estimators `ESN*/ELM*`: sklearn protocol, `get_params`/`set_params`,
  validation + `check_sequences` normalization, orchestration, and the single
  NumPy→torch (entry) / torch→NumPy (exit) conversion.
- Blocks `InputToNode`/`NodeToNode` (+variants) as configuration
  (hyperparameters, param round-trip). Standalone `block.transform(X_np)` still
  accepts/returns NumPy by converting at its own edge (back-compat).
- `model_selection`, `metrics`, `datasets`, `preprocessing` (Coates),
  `projection` — remain NumPy/sklearn.

**Backend (PyTorch, tensors — never imported by users)**
- Reservoir engine (batched recurrence), input feature-map compute, weight-init
  buffers, and the **readout** (ridge / incremental normal equations) — all in
  torch, on-device.
- `device`/`dtype` owned here (configured via estimator params).

**Consequences**
- Readout is torch by default (implements the `K`/`xTy` incremental normal
  equations + solve on-device, preserving the mergeable `__add__` semantics).
- The `regressor=` extension point: a torch-native readout is the default;
  passing a NumPy sklearn regressor is still allowed but forces a torch→NumPy
  round-trip at the readout (documented performance caveat). *(refine later)*
- Only two host↔device transfers per `fit`/`predict` call.

## State semantics (D6, D7)

**D6 — washout (training-only).** New estimator param `washout: int = 0`,
applied **per sequence**: during `fit`, skip the first `w` reservoir states and
targets from the readout least-squares accumulation. `predict` runs the full
length and returns all `T` steps (early steps warm-influenced but present, so
seq-to-seq length alignment is preserved). Default `0` = current behavior
(INV-1). Validate `w < min(Lᵢ)`. ELM has no temporal transient → washout N/A
(ignored / rejected for ELM).

**D7 — initial-state carry (backend primitive + modest frontend hook).**
- Backend reservoir primitive:
  `reservoir(X_t, initial_state=None) -> (states, final_state)` (tensors,
  on-device). `initial_state=None` ⇒ zero init (default, INV-1).
- Frontend: `predict(X, initial_state=None, return_state=False)`. With
  `return_state=True` returns `(y, final_state)`; `initial_state` seeds the
  reservoir to continue a series across calls. State crosses the estimator edge
  as NumPy (converted per D5); internal state is a device tensor.
- **Streaming deferred** (no `partial_predict`/`reset_state` yet) but the
  primitive above is the foundation, so it can be layered on without redesign.
- Additive, back-compatible API change (defaults preserve current behavior).

## Params / clone (D8)

**D8 — standard nested params, with a `**kwargs` convenience alias.**
- **Canonical params** are the sub-estimators (`input_to_node`, `node_to_node`,
  `regressor`) plus estimator-level params (`requires_sequence`,
  `decision_strategy`, `verbose`, `washout`, `device`, `dtype`, …). Addressed
  nested: `input_to_node__spectral_radius`. `get_params`/`set_params`/`clone`
  come from `BaseEstimator` recursing through each block's **public**
  `get_params` — we no longer *call* `_get_param_names` on sub-objects (the
  Phase-2 concern is gone).
- **`**kwargs` stays** as pure construction sugar so `ESNRegressor(
  hidden_layer_size=100)` keeps working; kwargs populate the default
  sub-estimators at build time and are NOT part of the param set.
- **Implementation note:** `**kwargs` in `__init__` makes sklearn's default
  `_get_param_names` raise (it forbids `VAR_KEYWORD`). So override our own
  `_get_param_names` (classmethod) to return the explicit param list; everything
  else (nested get/set, clone, GridSearchCV `input_to_node__…`) then works by
  standard sklearn machinery. Fixes the flat-namespace / bare-name `set_params`
  mismatch that caused the GridSearchCV friction.
- **Fallback** (if the override proves fragile): drop `**kwargs` entirely →
  pure nested, zero overrides, but loses the flat constructor shortcut.

## Training modes (D9)

**D9 — `nn.Module` architecture + two training paths (phased).** The backend is
built as PyTorch `nn.Module`s so both closed-form and gradient-based training of
RCNs are supported from one design.
- **Reservoir** = an `nn.Module`; its input/recurrent weights are `Parameter`s
  with **`requires_grad=False` by default** (classic fixed reservoir). The RC
  initialization strategy (spectral radius, `k_in`/`k_rec` sparsity, predefined
  weights) is preserved — it just fills these parameters. An opt-in toggle
  (`requires_grad=True`) makes the reservoir trainable end-to-end.
- **Readout** = an `nn.Linear`, populated one of two ways via a `solver`:
  - `closed_form` (default): analytically set weights via the ridge /
    normal-equations solve (the D5/D7 path). Requires a fixed reservoir.
  - `gradient`: train weights with an optimizer loop (works with a fixed or a
    trainable reservoir).
- **Valid configurations:** (1) fixed reservoir + closed-form (**default**,
  behavior-preserving); (2) fixed reservoir + gradient (converges to #1 for a
  linear readout under MSE — a built-in consistency check); (3) trainable
  reservoir + gradient. "trainable + closed-form" is invalid → validated out.
- **Gradient mode adds opt-in hyperparams:** optimizer, learning rate, epochs,
  loss, batch size. All default-off; closed-form users see no new required args.
- **Phasing:**
  - **Phase A** — torch backend, fixed reservoir + closed-form readout (exactly
    D1–D8, behavior-preserving), built on the `nn.Module` foundation so
    parameters / `requires_grad` exist from the start.
  - **Phase B** — add gradient training (trainable reservoir + optimizer loop);
    a new `fit` branch, no rewrite.
- Refines D2 (engine = `nn.Module`s), D5/D7 (readout is an `nn.Linear`).

**D11 — closed-form readout = `IncrementalRegression`, torch-native.** The
existing default readout class is reimplemented to run on torch tensors
(dual-mode: still accepts NumPy for standalone/public use, unchanged), so it
stays the default closed-form solver and computes on-device (filling the readout
`nn.Linear`'s weights via the ridge / normal-equations solve; mergeable `__add__`
preserved). A user-supplied external sklearn regressor (Ridge, …) still works in
the fixed-reservoir/closed-form path via a documented torch→NumPy round-trip;
it is not offered in gradient/trainable mode. (Gradient-mode readout = the
`nn.Linear` trained by the optimizer loop.)

## Block ↔ backend module, and dual usage (D12)

**D12 — companion pattern (not inheritance).** Frontend blocks
(`InputToNode`/`NodeToNode` + variants) stay sklearn `TransformerMixin`
**configuration** objects (hyperparameters, `get_params`/`set_params`). At
`fit`, each builds its backend torch `nn.Module` counterpart (dense
`Parameter`s + `forward`) that performs the computation. Blocks are **not**
`nn.Module` subclasses — that avoids the `nn.Module.__setattr__` vs sklearn
`get_params`/`clone` clash that made the old draft unworkable.

**Dual usage (enabled by D12) — an explicit goal.** Because the compute lives
in standalone torch `nn.Module`s, one codebase serves two audiences:
- **scikit-learn users** — the `ESN*/ELM*` estimators (NumPy I/O, full sklearn
  ecosystem), unchanged.
- **PyTorch users** — the backend `nn.Module`s used directly (tensors, custom
  training loops, composition into larger torch models).

**D13 — public torch API, staged, as `pyrcn.nn`.** The backend modules become a
*supported public* torch API **after Phase-A parity** (not before — Phase A
stays focused on correctness vs. the NumPy oracle). Namespace: **`pyrcn.nn`**,
built **clean from the companion modules**. ⚠️ Do **not** resurrect or port the
old `pyrcn.nn` draft (removed in commit `3e41df2`) — it was messy; the new
`pyrcn.nn` is a fresh, minimal surface over the Phase-A backend `nn.Module`s.

## Torch module implementation & reuse (D14)

**D14 — maximize reuse of `nn.RNNCell` / `nn.Linear`; add only what torch
lacks (leaky integration + extra activations).** No duplication of torch's
weight / init / `state_dict` / bidirectional machinery.

**Block → torch module mapping**
- **`InputToNode` → subclass `nn.Linear`** (`weight` = input weights, `bias`),
  plus `input_scaling`/`input_shift`/`bias_scaling`/`bias_shift` and the input
  activation applied in `forward`. This is the real feature map for **ELM**
  (ELM = these + readout) and the input stage for **ESN**.
- **`NodeToNode` → subclass `nn.RNNCell`** (recurrence). Reuse its fused
  single-step op for `tanh`/`relu` via `(1-λ)h + λ·super().forward(x, h)`
  (verified parity 1.1e-16). For `logistic`/`identity`/`bounded_relu` (not in
  the fused op) compute the affine with `F.linear` + apply the activation, still
  `+ leaky`. `weight_ih` = identity (input is added directly — the input weights
  belong to `InputToNode`); `spectral_radius` is folded into `weight_hh`. A thin
  layer loops the cell over time (lengths/mask, `initial_state`→`final_state`)
  and does bidirectional (reuse `nn.RNN` structure: tie reverse=forward+flip for
  the parity check, independent scaled weights otherwise).

**Two-activation handling.** PyRCN applies an *input* activation (`InputToNode`)
AND a *reservoir* activation (`NodeToNode`); a single RNN cell has one. So for a
nonlinear `input_activation` (e.g. the `ESNClassifier` default `tanh`)
`InputToNode` stays a separate stage feeding the cell (cell `weight_ih` =
identity). For `input_activation='identity'` the input weights may be folded into
the cell's `weight_ih` (a fused `ESNCell`).

**Efficiency notes / obstacles (all minor, none blocking).**
- `nn.RNN`'s *whole-sequence* fused kernel supports only non-leaky `tanh`/`relu`;
  leaky needs per-step control → per-step loop that still reuses the fused
  *single-step* cell op. An optional whole-sequence fast-path (delegating to
  `nn.RNN`) for the non-leaky `tanh`/`relu` case can be added later.
- The three extra activations use `affine + activation` rather than the fused
  cell op.

**Deferred:** a fused single-cell `ESNCell` (input+recurrence, one activation)
for `input_activation='identity'` as the idiomatic/fast path in the public
`pyrcn.nn`. Refactor note: the A2 custom `Reservoir` becomes a thin layer over
the `nn.RNNCell` subclass; its behavioral parity tests carry over unchanged.

## Roadmap / workstreams

Two phases. **Phase A** delivers the full torch backend with a fixed reservoir
and the closed-form readout — behavior-preserving (D1–D8, D10, D11), built on
the `nn.Module` foundation. **Phase B** adds gradient-based training (D9) as new
`fit` branches, no rewrite. The legacy NumPy path is kept importable until
Phase-A parity is proven, then retired.

### Phase A — torch backend, fixed reservoir + closed-form (behavior-preserving)
- **A0 · Scaffolding & deps. — done.** Add torch as a hard dependency (D4); create a
  private `pyrcn.backend` (torch) package; `device`/`dtype` helpers; stand up
  the parity harness (torch-float64 vs legacy NumPy-float64, D3). Legacy path
  stays alive for comparison.
- **A1 · Sequence normalization (D1 + task=). — done.** `check_sequences`: accept
  list / object-array / 2-D / 3-D → canonical padded batch `(N, L_max, F)` +
  `lengths`; task auto-detect + `task=` override; INV-1 (2-D → N=1). Replaces
  `concatenate_sequences` + `_check_if_sequence*`. Tests for every input form.
- **A2 · Reservoir engine (D2/D6/D7/D10). — done** (feature map, leaky /
  Euler / bidirectional cells, torch-native init; parity by injection).
  Reservoir `nn.Module`: dense
  `Parameter` input/recurrent weights (`requires_grad=False`), **RC init
  preserved** (spectral radius, `k_in`/`k_rec` sparsity, antisymmetric/Euler
  variants, predefined weights); batched masked recurrence over `L_max`; leaky
  integration; bidirectional; `washout`; `initial_state`→`final_state`. Input
  feature-map (`InputToNode` math) as an `nn.Module` too.
- **A3 · Closed-form readout (D11). — done.** `IncrementalRegression`
  reimplemented torch-native as `backend.IncrementalRidge`: incremental
  `K`/`xTy`, ridge solve on-device, `fit_intercept` via ones-column, mergeable
  `__add__` preserved. Parity vs the legacy readout by injecting identical
  features/targets (single-batch, postpone-then-solve, merge == concat).
- **A4 · Estimator integration (D5/D8). — core done.** `ESN*/ELM*`
  `fit`/`predict` route feature-map→reservoir→readout through the torch
  backend (single float64 conversion at the edge) via the config→backend
  bridge, on a **fast path + numpy fallback**: the fast path engages only when
  every component is a torch-backable pyrcn default; arbitrary sub-estimators
  (`FeatureUnion` input, external `Ridge`, `normalize=True`) and `partial_fit`
  keep the legacy numpy path. Param surface kept as-is (bare names) so
  model selection is unchanged (decision: keep bare, additive nested deferred).
  Classifier `predict_proba` + decision strategies preserved unchanged. Full
  suite green (incl. the formerly-slow chunk test, now ~13s on the fast path).
  A4b done: `washout` (int, training-only, drops start transient per
  sequence), `predict(initial_state=, return_state=)` (ESN-only, torch
  backend; state-carry primitive verified — split-and-carry reproduces the
  whole-sequence pass). Defaults behavior-preserving; numpy fallback raises
  NotImplementedError for these.
- **A5 · Parity & test migration. — done.** Broadened the torch-vs-numpy
  parity harness across all four estimators and both sequence/non-sequence
  modes via a native-vs-forced-fallback twin (fallback wraps the input in a
  single-transformer `FeatureUnion`); max observed difference ~3e-9 (float64
  round-off). Model-selection tests (GridSearchCV / RandomizedSearchCV /
  SequentialSearchCV / component-swap over bare-name params) confirm
  `best_estimator_._use_torch` stays True through `clone` — model selection
  unchanged (bare-name surface, per the A4 decision; the planned nested-params
  test is moot since we kept bare names). Dropped the joblib/loky
  sequence-parallel path (`n_jobs` now a serial no-op). `metrics` unaffected
  (predict still returns NumPy). Suite 194 passed / 2 skipped.
- **A6 · Finalize (legacy retirement deferred). — done.** Reframed: the numpy
  reservoir loop (`NodeToNode.transform`), `concatenate_sequences`, and the
  numpy weight init CANNOT be removed under fast-path + fallback — the fast
  path still calls `concatenate_sequences` and injects numpy-init weights, and
  the fallback path + standalone block tests + parity tests still exercise the
  numpy reservoir loop. True retirement waits until the companion pattern (P1+)
  makes the blocks torch-native. Done here: removed the dead+broken
  `__add__`/`__radd__` merge operators (only the dropped joblib `sum(reg)`
  used them; they read `regressor._K`, absent on a torch estimator) and two
  dead commented lines; added `tests/test_backend_device.py` (CPU runs, CUDA
  skipped when unavailable); dropped `joblib` from explicit deps (sklearn
  pulls it transitively). CI commands verified locally (flake8 `src/pyrcn
  tests`, mypy `src/pyrcn`, full pytest) — 196 passed / 3 skipped. User-facing
  docs deferred to P1 (backend still marked private).

### Public torch API — `pyrcn.nn` (after Phase-A parity, D13)
- **P1 · Promote the backend modules to a public, documented `pyrcn.nn`. —
  done.** Renamed `pyrcn.backend` → `pyrcn.nn` (single public package, built
  from the Phase-A companion modules — NOT the removed draft `3e41df2`).
  Public surface: reservoir layers `Reservoir`/`EulerReservoir`, cells
  `LeakyESNCell`/`EulerESNCell`, `InputFeatureMap`, `IncrementalRidge`;
  weight initializers under a `pyrcn.nn.init` submodule (mirrors
  `torch.nn.init`). Frontend glue (`_bridge`) and the cell base
  (`_ReservoirCell`) stay private. Removed the "implementation detail /
  not-public-yet" disclaimer; added a pure-PyTorch usage example
  (doctest-verified) and a Sphinx API page (`docs/.../pyrcn.nn.rst`).
  `pyrcn.nn` registered in the top-level package. No behavior change; suite
  195 passed / 3 skipped, flake8 + mypy clean.

### Phase B — gradient-based training (D9)
- **B1 · Gradient readout solver. — done.** Added `pyrcn.nn.LinearReadout`
  (trainable `nn.Linear` readout) + `pyrcn.nn.train_readout` (optimizer loop;
  optimizer/loss by name, `weight_decay` = ridge strength). `ESN*`/`ELM*`
  gain flat params `solver` (`closed_form` default / `gradient`), `optimizer`,
  `learning_rate`, `epochs`, `batch_size` (validated, in `get_params` for
  clone/GridSearch). In gradient mode the reservoir stays fixed: states are
  computed once (dropping `washout` per sequence, concatenated in sequence
  mode) and a fresh `LinearReadout` is trained; the `regressor` supplies the
  readout config (`fit_intercept`, `alpha`→`weight_decay`). Gradient requires
  native torch-backable components (else `NotImplementedError`). Consistency
  check (D9): on well-conditioned (whitened) features a gradient-trained
  readout matches the closed-form ridge to ~1e-6; through the raw reservoir /
  ELM map convergence is slow, so the estimator tests assert learning + the
  structural contracts. Defaults behavior-preserving; full suite green.
  Gradient fits are reproducible via `random_state`: the readout init and the
  shuffle are seeded from it (`pyrcn.nn.torch_generator`, threaded through
  `LinearReadout(generator=)` and `train_readout(generator=)`).
- **B2 · Trainable reservoir. — done.** `ESN*` gain `trainable_reservoir`
  (bool, default False). With `solver="gradient"` it unfreezes the reservoir's
  recurrent weights (`Reservoir/EulerReservoir.set_recurrent_trainable`) and
  optimizes them jointly with the readout by backprop through the recurrence
  (BPTT): each epoch recomputes the states (the fixed input feature map is
  applied once and detached), full-batch. The RC init is the starting point.
  The invalid `trainable + closed_form` combo is rejected; reproducible via
  `random_state`; `predict` runs under `no_grad`. ESN-only (ELM has no
  reservoir); sequence + non-sequence + classifier. Full suite 218 passed.
  - **B2+ (done).** `trainable_input` (bool): with `solver="gradient"`, also
    train the input feature-map weights (`InputFeatureMap.set_input_trainable`)
    — alone or with `trainable_reservoir` for a fully trainable RNN (input +
    reservoir + readout); requires the gradient solver. BPTT mini-batching:
    `batch_size` batches over sequences (seeded shuffle; `None` = full-batch),
    non-sequence stays full-BPTT. Reproducible via `random_state`. Suite 222.
- **B3 · Tests, examples, docs** for the gradient modes.

### Risks / notes
- NumPy-2 ragged/object-array handling lives only in `check_sequences`.
- float32 can diverge from float64 on long sequences → correctness proven in
  float64; float32 judged on task metrics (D3).
- Keep every change behind the preserved public API; the legacy path is the
  parity oracle until A5 passes.

## Open questions
- Target performance / scale (sequence count, lengths)?
- Any external backend dependency acceptable (numba, Cython), or pure
  NumPy/SciPy only?
