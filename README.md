# NeuroFence

**LLM Weight Poisoning & Backdoor Scanner — offline forensic tool**

An offline AI-security / model-forensics tool. It loads a local open-source LLM
inside a sandbox, records the model's identity and a SHA-256 fingerprint,
captures internal Transformer activations via PyTorch forward hooks, and writes
activation statistics to JSON and CSV.

**Phase 1** builds the evidence substrate: sandboxed loading, fingerprinting,
activation capture, statistics. **Phase 2** adds adversarial fuzzing, baseline
comparison, anomaly detection, trigger-consistency measurement, and risk
scoring — validated against a controlled experiment with known ground truth.

The detector reports **evidence strength, not probability of compromise**. It
measures activation anomalies; it does not establish intent, and it cannot tell
you a model is safe.

---

## Pipeline

```
Local LLM
   ↓
Model Sandbox          offline env, socket blocking, no autograd
   ↓
Model Metadata         architecture, layers, parameters, versions
   ↓
SHA-256 Fingerprint    per-file digests + combined model/weights hashes
   ↓
Prompt Fuzzer          6 categories, reproducible, deduplicated   [Phase 2]
   ↓
Baseline Activation    per-layer distributions from NORMAL prompts [Phase 2]
   ↓
Test Activation        forward hooks on transformer blocks
   ↓
Anomaly Detection      robust z-scores vs baseline                 [Phase 2]
   ↓
Trigger Consistency    across carriers, against control tokens     [Phase 2]
   ↓
Risk Score             weighted, explainable, 0-100                [Phase 2]
   ↓
Security Findings      JSON / CSV Results
```

---

## Requirements

- Python 3.11+
- PyTorch 2.2+ (CPU is fine; CUDA used automatically if present)
- ~2 GB RAM for a 0.5B model at float32

## Installation

```bash
git clone <your-repo-url> NeuroFence
cd NeuroFence

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install torch first if you need a specific build:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

---

## Quick start

### 1. Stage a model

Scanning **never** downloads. Staging is a separate, explicit step:

```bash
# Option A — download from the Hub (needs network + huggingface-hub)
pip install huggingface-hub
python scripts/prepare_model.py --model Qwen/Qwen2.5-0.5B-Instruct --download

# Option B — copy an existing local checkout
python scripts/prepare_model.py --source /path/to/model --dest models/base

# Option C — no model handy? Build a tiny random one to smoke-test the pipeline
python scripts/make_test_model.py --dest models/test-tiny
```

Staging writes `neurofence_fingerprint.json` into the model directory and
prints the SHA-256 values.

### 2. Inspect it

```bash
python scripts/inspect_model.py
python scripts/inspect_model.py --model-path models/test-tiny
python scripts/inspect_model.py --model-path models/test-tiny --json
```

Read-only: loads the model, prints metadata, the fingerprint, and exactly where
hooks will be placed. No inference.

### 3. Run the full Phase 1 pipeline

```bash
python scripts/inspect_model.py --model-path models/base --run
```

Or from Python:

```python
from neurofence import run_phase1

result = run_phase1(prompts=["Explain what a firewall does."])
print(result.run_dir)
print(result.loaded.fingerprint.model_sha256)
```

---

## Output

Each run creates its own directory under `data/`:

| File | Contents |
|---|---|
| `manifest.json` | Model identity, fingerprint, full config, sandbox report |
| `metadata.json` | Architecture, layers, parameters, library versions |
| `fingerprint.json` | Per-file SHA-256 digests, combined hashes |
| `activations.json` | Aggregate **and per-prompt** statistics per capture site |
| `activations.csv` | Flat aggregate table for pandas/Excel |

Logs go to `logs/<run_name>.jsonl` as JSON Lines.

### Statistics captured per site

Descriptive only — no thresholds, no scoring.

`mean` · `std` · `variance` · `min` · `max` · `max_abs` · `mean_abs` · `rms` ·
`l2_norm` · `mean_token_l2` · `skewness` · `kurtosis` · `sparsity` ·
`outlier_ratio` · `top_channels`

`kurtosis` and `outlier_ratio` are the fields most likely to matter downstream:
the weight-poisoning literature repeatedly finds backdoors expressed as a few
extreme channels rather than a shift in the global mean. `top_channels` records
which channels those were.

---

## Configuration

All behaviour is driven by `config/default.yaml`. Every key has a documented
default; unknown keys are **rejected**, not ignored (a typo in a security
tool's config should fail loudly).

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  local_path: "./models/base"
  device: "auto"              # auto | cpu | cuda
  dtype: "float32"
  trust_remote_code: false

sandbox:
  allow_network: false
  block_sockets: true

activation:
  capture_points: [block_output]   # + attention_output, mlp_output
  layers: "all"                    # or "0-5,10,23"
  max_new_tokens: 0                # 0 = forward pass only
  top_k_outliers: 5
```

Override anything from the environment:

```bash
NEUROFENCE_MODEL__DEVICE=cpu NEUROFENCE_RUN__SEED=42 python scripts/inspect_model.py --run
```

---

## Design decisions

**Two fingerprints.** `weights_sha256` covers weight files only;
`model_sha256` covers config and tokenizer too. Config and tokenizer tampering
is a real backdoor vector, but it shouldn't falsely implicate the weights, so
the two are reported separately.

**Per-prompt statistics, not just aggregates.** A trigger-activated backdoor
disappears in a run-wide average. Phase 2 needs per-prompt breakdowns to
compare trigger candidates against a clean baseline.

**Float64 throughout the statistics layer.** Squaring float16 activations
overflows, and a silent `inf` variance looks exactly like an anomaly.
Everything is promoted to float64 on CPU before accumulation.

**Streaming accumulation.** Statistics fold in incrementally, so a run over
many prompts never holds all activations in memory.

**Architecture-agnostic hook placement.** Known paths cover Llama, Qwen,
Mistral, Gemma, Phi-3, GPT-2, GPT-NeoX, Falcon, OPT, and MPT. Anything else
falls back to a structural search for the largest uniform `ModuleList` — a
poisoned model is not obliged to be a Llama.

**Downloads are a separate process.** A scan that can fetch fresh weights
defeats the fingerprint, because "the file changed under us" is precisely what
the fingerprint exists to detect.

---

## Phase 2 — running a scan

```bash
python scripts/scan.py --model-path models/base
python scripts/scan.py --model-path models/base --trigger PINEAPPLE --json report.json
```

Or from Python:

```python
from neurofence.phase2 import run_phase2

result = run_phase2()
print(result.trigger_result.verdict)   # e.g. "no_trigger_behaviour"
print(result.risk.score)               # 0-100
```

### Verdicts

The scan returns one of these, never a bare yes/no:

| Verdict | Meaning |
|---|---|
| `trigger_behaviour_detected` | Anomaly reproduced across carriers **and** exceeded controls |
| `novelty_not_trigger` | Anomaly present, but control tokens produced a comparable one |
| `inconsistent_effect` | Beat the controls but did not reproduce across carriers |
| `no_trigger_behaviour` | No effect above threshold |
| `inconclusive_no_controls` | Controls were not run; no verdict is possible |
| `measurement_unreliable` | Determinism check failed; all results suspect |
| `insufficient_evidence` | No trigger prompts were scored |

Note that four of the seven outcomes are refusals to conclude. That is
deliberate.

---

## Controlled experiment

The detector is validated against known ground truth, the way an antivirus is
tested with an EICAR file rather than live malware:

```bash
python scripts/run_experiment.py --source models/base --workdir data/experiment
```

This builds two arms — an unmodified **clean** copy (known negative) and a
**poisoned** copy (known positive) — scans both, and reports the confusion
matrix. Exit code is 0 on pass, 2 on a false positive or false negative.

Representative run on the bundled fixture model:

```
  arm        expected   verdict                            risk
  --------------------------------------------------------------
  clean      negative   no_trigger_behaviour               15.5
  poisoned   positive   trigger_behaviour_detected         87.9

  TP=1  FP=0  TN=1  FN=0
  Risk gap (poisoned - clean): 72.4
  RESULT: PASS
```

### What the planted modification is

It scales the input-embedding row for the trigger token, producing a larger
activation signature when that token appears. It is an **activation marker
only**: it teaches the model nothing, changes no outputs, and is not a
functional backdoor. Its purpose is to test the measurement instrument.

Safety properties: never modifies a model in place, always writes to a separate
destination, and drops a `BACKDOOR_README.txt` plus a manifest into the output
so a test artefact can never be mistaken for a clean model.

---

## Why the controls matter

A rare token produces unusual activations simply by being rare. Every trigger
prompt is therefore matched against control tokens (APPLE, BANANA, ORANGE,
MANGO) run through the same carrier sentences. If the controls score as
anomalously as the trigger, the honest conclusion is *"this token is unusual"*,
not *"this model is backdoored"* — and the tool says so.

The detector **cannot reach a positive verdict without the control comparison
having been run.**

### On "run the trigger five times"

A forward pass with no sampling is deterministic: the same string gives
bit-identical activations every time. Repeating one prompt therefore measures
nothing about a backdoor. So consistency is measured two ways:

- **determinism** — identical repeats, which must be exactly 1.0. This checks
  the *harness*, not the model. If it fails, every other number is suppressed.
- **consistency** — across distinct carrier sentences containing the trigger.
  This is the real signal: does the anomaly follow the token, or the sentence?

---

## Detection configuration

```yaml
fuzzer:
  seed: 42
  normal_prompts: 100          # builds the baseline
  trigger_prompts: 50
  control_prompts_per_token: 25
  trigger: "PINEAPPLE"
  control_tokens: ["APPLE", "BANANA", "ORANGE", "MANGO"]
  determinism_repeats: 5
  log_prompt_text: false       # prompt text stays out of logs and reports

detection:
  method: "robust"             # median/MAD; falls back to mean/std if MAD is 0
  threshold: 3.0
  min_consistency: 0.6         # trigger must fire on this fraction of prompts
  min_separation: 2.0          # ...and exceed controls by this margin
  weight_trigger_consistency: 0.35
  weight_control_separation: 0.35
  weight_layer_concentration: 0.15
  weight_anomaly_magnitude: 0.15
```

Every threshold and weight is configurable; none are hard-coded. Weights must
sum to 1.0 or the config refuses to load. The risk report always includes its
own components and weights so a reader can reconstruct the score.

---

## Design decisions (Phase 2)

**One prompt is one observation.** Baselines are distributions over *prompts*,
not over raw tensor elements. Pooling elements would give millions of samples
and a vanishingly small standard deviation, against which every input scores as
wildly anomalous — impressive-looking z-scores that mean nothing.

**Zero variance is undefined, never infinite.** A layer whose baseline never
varies cannot produce a z-score. Reporting `inf` there is the easiest way for
this tool to produce confident nonsense, so such layers are marked undefined
with a recorded reason.

**Robust by default.** Median/MAD rather than mean/std, because a few unusual
baseline prompts inflate the standard deviation and mask real anomalies. When
MAD collapses to zero (more than half the samples identical) it falls back to
mean/std, and records `zscore_mad_fallback` in the output so the substitution
stays auditable.

**Independent RNG streams per category.** Changing `trigger_prompts` leaves
every NORMAL prompt byte-identical, so two runs with different settings remain
comparable.

**Weak baselines cap the score.** A baseline below 20 prompts caps risk at 40;
missing controls cap it at 30; a failed determinism check zeroes it.

---

## Desktop application

```bash
pip install -e ".[gui]"
python scripts/desktop.py
```

A local PyQt6 application — no web server, no browser, no cloud APIs. Six
screens: **Dashboard**, **Model**, **Configure**, **Scan**, **Findings**,
**Visualisations**.

Scanning runs on a `QThread` and reports named stages (`Loading model...`,
`Calculating fingerprint...`, `Preparing prompts...`, and so on) rather than a
smoothly-advancing percentage, because stage durations are wildly uneven and a
smooth bar would be a lie. The GUI stays responsive throughout; cancellation is
cooperative and checked between stages, since a forward pass cannot be
interrupted mid-flight without corrupting hook state.

Before a scan runs, every field shows an em-dash rather than zero — *"0
anomalies"* and *"not yet measured"* mean very different things.

The Configure screen validates every input and refuses to start a scan that
cannot produce a meaningful result: no control tokens, a trigger listed as its
own control, or a baseline under 20 prompts are all rejected with an
explanation rather than silently producing a confident-looking number.

---

## PDF reporting

```bash
python scripts/scan.py --model-path models/base --pdf report.pdf
```

Or **File → Export PDF report…** in the desktop app.

The report contains: cover, executive summary, model information, scan
configuration, findings, activation analysis with real figures, the heatmap,
risk assessment, experimental evaluation, recommendations, and limitations.

Three rules are enforced structurally rather than by convention:

- **Limitations are always emitted**, including on clean reports. A reader who
  sees only good news must still see what the tool did not test.
- **No figure without data.** Charts render only from measured results; a chart
  whose data is absent is omitted, never replaced with a placeholder.
- **Fixed vocabulary.** Verdicts map to a fixed label table, so no call site can
  invent stronger wording than the evidence supports. `trigger_behaviour_detected`
  renders as *"Potential Backdoor Indicator"* — never "Confirmed Malware".
  The terms *confirmed*, *malware*, *malicious*, and *compromised* appear
  nowhere in the report vocabulary, and a test asserts it.

Experimental evaluation (detection rate, false positive rate, precision,
recall, F1) appears **only** when a controlled experiment was actually run.
Otherwise the section states plainly that no such figures are available, rather
than printing metrics that were never measured.

---

## Visualisations

Five charts, all from measured data:

| Chart | Shows |
|---|---|
| Activation heatmap | Prompt category × transformer layer |
| Baseline vs trigger | Observed activation against the baseline mean |
| Suspicious layers | Per-layer peak score with the threshold marked |
| Trigger vs controls | The core validity check, side by side |
| Risk breakdown | Each component's weighted contribution |

Plus **clean vs poisoned** when a controlled experiment is supplied.

The heatmap switches to a logarithmic colour scale when the dynamic range
exceeds 20×, and says so in its caption. A strongly-anomalous trigger row is
often 30–40× the others; on a linear scale that crushes every other row to
solid black and hides real structure. The trigger still reads as brightest —
the other categories simply become legible.

---

## Repository layout

```
NeuroFence/
├── config/default.yaml
├── src/neurofence/
│   ├── core/          config, logging, exceptions
│   ├── model/         loader, adapter, metadata, hashing
│   ├── sandbox/       sandbox, policy
│   ├── activation/    hooks, collector, statistics, storage
│   ├── fuzzing/       generator, categories, corpus, seed      [Phase 2]
│   ├── baseline/      builder, analyzer                        [Phase 2]
│   ├── detection/     anomaly, trigger_analysis, risk_score    [Phase 2]
│   ├── experiments/   controlled_backdoor, clean_model, validation
│   ├── reporting/     schemas, report, pdf                     [Phase 3]
│   ├── visualization/ charts                                   [Phase 3]
│   ├── desktop/       app, screens, worker, theme              [Phase 3]
│   ├── pipeline.py    Phase 1 orchestration
│   ├── phase2.py      Phase 2 orchestration
│   └── cli.py
├── scripts/
│   ├── prepare_model.py     stage + fingerprint (only network-capable script)
│   ├── inspect_model.py     read-only inspection, or --run for Phase 1
│   ├── scan.py              scan, with --pdf reporting
│   ├── desktop.py           launch the desktop application
│   ├── run_experiment.py    controlled backdoor validation
│   └── make_test_model.py   fixture model for offline testing
├── tests/
│   ├── unit/                215 tests
│   └── integration/         37 tests (marked slow)
├── models/            staged models (gitignored)
├── data/              scan output (gitignored)
└── logs/              JSONL logs (gitignored)
```

---

## Testing

```bash
pytest                        # everything
pytest -m "not slow"          # unit tests only, no model loading
pytest tests/integration -v   # full pipeline against a tiny real model
ruff check .
```

**252 tests, all passing.** The suite builds genuine tiny Llamas (real
safetensors, real tokenizer, random weights) so everything runs **offline**
with no downloads.

GUI tests run under Qt's `offscreen` platform, so the whole suite is headless. One of them starts a real scan and runs a 50 ms heartbeat timer alongside it, asserting the timer keeps firing — a direct check that the event loop is never blocked.

The load-bearing test is the controlled experiment: it plants a known backdoor,
scans both arms, and asserts zero false positives and zero false negatives.
There is also a negative-control test that asks about the *wrong* trigger on the
poisoned model and asserts it is not detected — guarding against a detector that
simply flags whatever token it is handed.

---

## Scope and limitations

- **You supply the trigger.** NeuroFence tests candidate triggers; it does not
  search the space of possible triggers. A negative result means *this trigger
  was not found*, not that the model is clean.
- **The sandbox is a guardrail, not a security boundary.** See
  [SECURITY.md](SECURITY.md) before scanning weights you actually distrust.
- **Validated only on small fixture models.** The controlled experiment passes
  on a 6-layer random-weight Llama with an obvious planted marker. A real
  backdoor in a production model will be subtler, and detection performance
  there is unmeasured.
- **Only Llama-family architectures have been run end to end.** The adapter has
  probe paths for GPT-2, NeoX, Falcon, OPT, and MPT with unit coverage, but no
  real model of those families has been loaded through it.
- **Activations only.** Weight-space analysis, gradient-based trigger search,
  and output-behaviour testing are all out of scope.

---

## Roadmap

- **Phase 1 — Foundation** *(complete)* — sandbox, metadata, fingerprinting,
  activation capture, statistics, storage.
- **Phase 2 — Detection** *(complete)* — adversarial fuzzing, baselines, anomaly
  detection, trigger consistency, risk scoring, controlled validation.
- **Phase 3 — Forensic application** *(complete)* — PyQt6 desktop app,
  activation visualisations, PDF reporting, experimental evaluation.

Future work: automated trigger search, weight-space analysis, and case
management across multiple scans.

## License

MIT — see [LICENSE](LICENSE).
