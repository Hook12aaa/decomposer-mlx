# Decomposer Auto-Researcher — Design Spec

**Date:** 2026-05-18
**Status:** Approved for implementation planning
**Project name:** `decomposer.research` (subpackage of existing decomposer project)

## Goal

Cut decomposer inference latency from **~12 minutes per run** to **~30 seconds per run** (target: 20×–24× speedup) by running a disciplined, semi-autonomous optimization loop that experiments on the existing PyTorch+MPS implementation. Validates each optimization against the X-ray trace (objective: stage-level wall_ms) AND a pixel-similarity quality oracle (subjective: layer decomposition fidelity).

The auto-researcher reuses the X-ray infrastructure already in `decomposer.core.xray` — every run produces a trace; experiments are scored by `diff-traces` against a pinned baseline plus an SSIM-based oracle on output layers.

## Non-goals (this phase)

- **MLX port for fused dequant kernels** — the research found that the true latency ceiling is the per-call GGUF dequant overhead (we're ~1000× off the M3 Max bandwidth ceiling). Achieving the stretch target of 10s requires porting the DiT to MLX with native fused 4-bit matmul kernels (mflux template). That's a 2-6 week manual project, scoped separately as a v3 spec. The auto-researcher is the v2 phase that lets us re-evaluate MLX with real-world delta data.
- **Quality oracle via VLM judge** — VLM scoring is expensive and subjective. Initial oracle is pure SSIM + non-degeneracy checks. VLM spot-check is a manual audit on MERGEd experiments, not in the auto-loop.
- **Distributed experiment running** — single-machine, one experiment at a time. Worktree isolation allows parallelism, but the auto-researcher v1 runs sequentially to keep MPS-OOM blast radius small.
- **Training new LoRAs / distillation** — only existing published checkpoints in the experiment queue. Custom Lightning distillation is a separate $300 RunPod project.

## The critical insight that shapes this design

From the 2026-05-18 optimization research:

> "M3 Max bandwidth: ~300 GB/s sustained. 21 GB Q8 DiT → bandwidth floor per step: ~70 ms. We observe: 77,000 ms per step. **We're ~1000× off the bandwidth ceiling.** The per-call Python dequant in `GgufLinear.forward` is the bottleneck."

**Implication: dropping Q8 → Q4 saves MEMORY but not LATENCY.** Quantization further doesn't fix the per-step time; it only enables architectural wins (keep-warm). True latency reduction at this tier comes from: (a) fewer steps (Lightning), (b) caching (FBCache), (c) eliminating overhead (keep-warm, pre-allocation), (d) scheduler/sampler choice (UniPC).

The auto-researcher's job is to **test the cheap-to-try optimizations and quantify their stacked impact** before we commit to the expensive MLX port.

## Architecture

```
decomposer/
└── research/
    ├── __init__.py
    ├── baseline.py        # capture & pin reference run
    ├── experiments.py     # Hypothesis dataclass + queue loader
    ├── oracle.py          # SSIM + composite + non-degeneracy + Hungarian matching
    ├── runner.py          # worktree dispatcher, run, score, decide
    ├── ledger.py          # append-only JSONL of every experiment outcome
    ├── report.py          # human-readable summary of ledger
    └── cli.py             # `decomposer research <subcommand>` entrypoints

docs/superpowers/research/
├── queue.yaml             # initial experiment queue (tier 1)
└── results/               # per-experiment markdown reports (auto-written)

runs/
├── baseline-<timestamp>/
│   ├── trace.json
│   ├── trace.perfetto.json
│   ├── layer_0.png ... layer_N.png
│   └── composite.png      # alpha-composited reconstruction
└── ledger.jsonl           # append-only experiment record
```

## CLI surface

```bash
decomposer research baseline [--image PATH] [--layers N] [--steps N] [--resolution R]
   # Run the current main-branch impl on the reference image; pin to runs/baseline-<ts>/
   # Sets the "no-regression" target for all subsequent experiments.

decomposer research run \
   --queue docs/superpowers/research/queue.yaml \
   [--budget 8h] [--target-latency 30s] [--max-experiments 10]
   # Execute experiments from the queue. Stops on any of:
   #   - budget exhausted (wall time)
   #   - target latency hit
   #   - max-experiments hit
   #   - 3 consecutive non-merging experiments (ROI flat)

decomposer research report
   # Human-readable summary of ledger.jsonl:
   #   - Stacked wins (what merged, total speedup)
   #   - Discarded experiments + reasons
   #   - Remaining queue items

decomposer research replay <exp-id>
   # Re-run a specific experiment from the ledger manually for inspection.

decomposer research diff-runs <run-a> <run-b>
   # Side-by-side comparison of two completed experiment runs.
   # (Existing `decomposer diff-traces` underneath.)
```

## Experiment hypothesis schema (`queue.yaml`)

```yaml
experiments:
  - id: lightning-lora-4step
    description: "Load lightx2v Lightning-8steps LoRA, run at 4 steps"
    apply:
      kind: lora_load
      repo: lightx2v/Qwen-Image-Lightning
      filename: Qwen-Image-Lightning-8steps-V2.0.safetensors
      scale: 1.0
    overrides:
      steps: 4
      true_cfg_scale: 1.0
    predicted_delta:
      denoise_loop.wall_ms: -50%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85

  - id: q5km-dit
    description: "DiT to Q5_K_M GGUF (extend GgufLinear)"
    apply:
      kind: code_patch
      patches:
        - port_q5km_dequant_to_ggufloader
        - swap_gguf_file_to_q5km
    predicted_delta:
      load_dit.mps_alloc_peak_mb: -7000
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85
    # ... etc
```

Hypothesis kinds:
- `lora_load` — pipe.load_lora_weights(...)
- `code_patch` — apply named patches from `decomposer/research/patches/` to the worktree
- `env_var` — set DECOMPOSER_* var
- `setting_change` — modify Settings field
- `scheduler_swap` — change FlowMatchEulerDiscreteScheduler to UniPC/etc.

## Quality oracle (`oracle.py`)

```python
@dataclass
class QualityReport:
    composite_ssim: float
    per_layer_ssim_matched: float
    per_layer_ssim_individual: list[float]
    layer_match_indices: list[int]  # Hungarian matching: experiment[i] ↔ baseline[match[i]]
    non_degenerate: bool
    degeneracy_reasons: list[str]
    notes: list[str]

    def passes(self) -> bool:
        return (
            self.non_degenerate
            and self.composite_ssim >= 0.92
            and self.per_layer_ssim_matched >= 0.85
        )


def score(
    experiment_layers: list[PIL.Image.Image],
    baseline_layers: list[PIL.Image.Image],
    input_image: PIL.Image.Image,
) -> QualityReport: ...
```

Implementation notes:

1. **Composite SSIM**: alpha-composite experiment layers into a single RGB image; SSIM against `input_image` resized to match. This catches "did the output get visibly worse?"
2. **Per-layer SSIM with Hungarian matching**: compute pairwise SSIM matrix between all experiment layers and all baseline layers; use `scipy.optimize.linear_sum_assignment` to find the best permutation; report the mean SSIM under that matching. This catches "did the per-layer content change?" while correctly handling layer reordering.
3. **Non-degeneracy**: each layer must have ≥1% opaque pixels AND ≥1% transparent pixels (configurable thresholds). Catches "experiment produced empty / fully-opaque layers."
4. **No batching across runs** — each experiment is scored independently against the same baseline. The baseline is pinned once at `research baseline` time and not updated when experiments MERGE (it's the "v1 reference" forever within a research session). New baselines are explicit user actions.

Thresholds (conservative, per user decision):
- `composite_ssim_min = 0.92`
- `per_layer_ssim_min = 0.85`
- `min_opaque_fraction = 0.01`
- `min_transparent_fraction = 0.01`

## Decision logic (`runner.decide`)

```python
def decide(quality: QualityReport, perf: PerfReport, hypothesis: Hypothesis) -> Decision:
    if not quality.non_degenerate:
        return Decision.REJECT_DEGENERATE
    if quality.composite_ssim < hypothesis.composite_ssim_min:
        return Decision.REJECT_QUALITY
    if quality.per_layer_ssim_matched < hypothesis.per_layer_ssim_min:
        return Decision.REJECT_QUALITY
    if perf.total_wall_ms_delta > 0.05:  # ≥5% slower
        return Decision.REJECT_REGRESSION
    if perf.total_wall_ms_delta < -0.03:  # ≥3% faster
        return Decision.MERGE
    return Decision.KEEP_FOR_REVIEW  # near-neutral, kept but not merged
```

Decisions:
- **MERGE**: promote the worktree to main; this experiment's changes become the new baseline for all subsequent experiments in the queue
- **REJECT_***: archive the worktree under `worktrees/archive/`; record reason in ledger
- **KEEP_FOR_REVIEW**: archive the worktree under `worktrees/review/`; surface in the final report for human decision

## Worktree mechanics

Uses the `superpowers:using-git-worktrees` skill pattern:

```
worktrees/
├── exp-lightning-lora-4step/    # active or archived
├── exp-q5km-dit/
├── exp-bnb-text-encoder/
└── archive/
    ├── exp-failed-fbcache/      # rejected, kept for retrospective
    └── ...
```

Each experiment:
1. Branch from current main: `git worktree add worktrees/exp-<id> -b research/exp-<id>`
2. Apply hypothesis (one of the `apply.kind` types above)
3. Run `decomposer decompose <reference_image> --layers <N> --steps <N> --out worktrees/exp-<id>/run/`
4. Score quality + diff perf
5. Decide
6. If MERGE: `git merge research/exp-<id>` into main; subsequent experiments branch from the new main
7. If REJECT/KEEP: move worktree to `worktrees/archive/` or `worktrees/review/`

## Ledger format (`ledger.jsonl`)

```json
{
  "timestamp": "2026-05-18T15:42:00Z",
  "experiment_id": "lightning-lora-4step",
  "hypothesis": { /* full Hypothesis serialization */ },
  "worktree_path": "worktrees/exp-lightning-lora-4step",
  "baseline_run_id": "baseline-1779111649",
  "experiment_run_id": "exp-1779118921",
  "perf": {
    "baseline_total_wall_ms": 727027,
    "experiment_total_wall_ms": 361002,
    "delta_pct": -50.3,
    "stage_deltas": {
      "denoise_loop": {"baseline_ms": 616009, "experiment_ms": 308004, "delta_pct": -50.0}
    }
  },
  "quality": {
    "composite_ssim": 0.943,
    "per_layer_ssim_matched": 0.881,
    "non_degenerate": true,
    "notes": []
  },
  "decision": "MERGE",
  "merged_commit_sha": "abc123",
  "human_audit_pending": true
}
```

`research report` reads this file end-to-end.

## Initial tier 1 experiment queue

10 experiments seeded from the research findings, in order:

1. **`lightning-lora-4step`** — `lightx2v/Qwen-Image-Lightning-8steps-V2.0`, 4 steps, CFG=1
2. **`unipc-scheduler`** — UniPCMultistepScheduler at 8 steps
3. **`bnb-text-encoder`** — `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`
4. **`q5km-dit`** — port Q5_K_M dequant to GgufLinear, swap GGUF file
5. **`q4km-dit`** — same but Q4_K_M (more aggressive)
6. **`keep-warm`** — refactor MpsBackend to hold all 3 models in memory permanently (requires Q4 to fit budget)
7. **`fb-cache-8step`** — `diffusers.hooks.apply_first_block_cache` at threshold 0.08 (only valid if 8 steps, not stacked with Lightning)
8. **`pre-allocate-latents`** — reuse latent tensor across denoise loop
9. **`torch-compile-probe`** — `torch.compile(transformer)` with graph-break tolerance, measure
10. **`mps-empty-cache-tuning`** — `torch.mps.empty_cache()` placement between stages

Stretch (added if tier 1 doesn't reach 30s):
11. **`mixed-precision-ffn`** — torchao INT8 weight-only on FFN layers
12. **`lower-resolution-bucket`** — investigate if 512px bucket is supported (currently 640/1024 only)

## Quality regression handling

If an experiment MERGEs and a later experiment fails to MERGE for quality reasons that trace to the earlier merge (e.g., Lightning LoRA makes FBCache thresholds untenable), the ledger captures both events. `research report` flags suspected interactions.

A human audit queue (`runs/audit/`) collects every MERGE for VLM or manual review. The auto-researcher does NOT block on audit; audits happen async. If an audit fails, the user runs `decomposer research revert <experiment-id>` to roll back.

## Stopping criteria

The auto-researcher stops on any of:

1. Wall time budget exhausted (`--budget`)
2. Target latency reached (`--target-latency`) — the most recent successful baseline's `total_wall_ms` < target
3. Max experiments executed (`--max-experiments`)
4. 3 consecutive experiments fail to MERGE (ROI flat — diminishing returns)
5. Queue exhausted

## Failure handling

Each experiment runs in a subprocess via `subprocess.run` with a 1800s timeout. Failures captured in the ledger:

- **MPS_OOM**: torch.cuda.OutOfMemoryError or MPS allocator failure; experiment rejected, ledger records the stage at which it OOM'd
- **MODEL_DOWNLOAD_FAILED**: HF auth or network error; reject with retry suggestion
- **TIMEOUT**: experiment didn't complete within budget; reject
- **EXCEPTION**: any other Python exception; reject with traceback logged

## Tech additions

| Dep | Why |
|---|---|
| `scikit-image` | SSIM implementation |
| `lpips` | optional LPIPS perceptual loss (Tier 1 may be SSIM-only) |
| `scipy` | Hungarian matching (`linear_sum_assignment`) |
| `pyyaml` | queue.yaml parsing |

## Build order (for the implementation plan)

1. **`oracle.py`** + tests with synthetic images (CPU-only, fast iteration)
2. **`baseline.py`** + CLI `decomposer research baseline` (captures current state)
3. **`experiments.py`** + queue.yaml schema + `decomposer research load-queue` (validation)
4. **`runner.py`** worktree dispatch + decision logic (mock backend for testing)
5. **`ledger.py`** + `decomposer research report`
6. **CLI integration** — wire `research` subcommand into existing typer app
7. **Apply hypotheses** — implement `lora_load`, `code_patch`, `env_var`, `setting_change`, `scheduler_swap`
8. **Patches library** — `decomposer/research/patches/` for code-patch hypotheses (port_q5km_dequant_to_ggufloader, etc.)
9. **Initial queue.yaml** — write the 10 tier 1 experiments
10. **Execute** — actually run `decomposer research run` against tier 1; iterate based on real results

## Open risks

| Risk | Mitigation |
|---|---|
| SSIM oracle misses semantic decomposition failures (layer merging) | Composite SSIM check + Hungarian matching + VLM spot-audit on every MERGE |
| Worktree disk usage explodes (each experiment ~50 GB if it re-downloads weights) | Share HF cache via `HUGGINGFACE_HUB_CACHE` env var across worktrees; only patches/code diverge |
| `lightx2v/Qwen-Image-Lightning` doesn't transfer to Qwen-Image-Layered | Decline gracefully on quality oracle reject; ledger captures attempt; flag for $300 custom-distillation decision |
| MERGE breaks something later in the queue | KEEP_FOR_REVIEW captures borderline cases; report surfaces interaction failures; user can `revert` any merge |
| Auto-researcher runs unattended overnight and burns through HF rate limits | Budget cap; cached weights are reused; experiments serialized so concurrent rate consumption is bounded |
| MLX port becomes the only remaining win and we've over-invested in this phase | Honest framing in spec; auto-researcher answers "how much headroom is left?" — if tier 1 lands at 25s, MLX may be unnecessary; if at 45s, MLX is justified |

## Done criteria

- `decomposer research baseline` captures a reference run
- `decomposer research run` executes the 10-experiment queue and produces a ledger
- `decomposer research report` summarizes wins, regressions, and remaining headroom
- All quality oracle code is unit-tested against synthetic images (CI green)
- A final markdown report at `docs/superpowers/research/tier1-results.md` shows the stacked speedup achieved and identifies the next bottleneck (likely "per-step dequant — needs MLX")
