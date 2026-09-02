# Security

NeuroFence loads model files that may be deliberately malicious. Read this
before pointing it at weights you do not trust.

---

## The sandbox is a guardrail, not a security boundary

`neurofence.sandbox` runs **inside the same Python process** as the model. It
patches `socket.socket`, sets offline environment variables, and disables
autograd. That stops the tool and well-behaved libraries from phoning home or
executing remote code *by accident*.

It does **not** contain deliberately malicious code. Anything running in-process
can undo an in-process restriction:

| Attack | Stopped by the sandbox? |
|---|---|
| Library silently fetching from the Hub mid-scan | Yes |
| `trust_remote_code` executing model-supplied Python | Yes — blocked unless explicitly opted in |
| Accidental outbound telemetry | Yes |
| Pickle deserialisation RCE from a `.bin` file | **No** — code runs at load time |
| Compiled native extension shipped with the model | **No** |
| Malicious code restoring the original `socket.socket` | **No** |
| Filesystem reads/writes outside the repo | **No** |

**For genuinely untrusted weights, run NeuroFence inside a VM or container with
no network interface and no access to anything you care about.** The in-process
sandbox is defence in depth, not the primary control.

---

## Threat model

**What NeuroFence assumes.** The model files are potentially hostile. The host
running NeuroFence is trusted. The Python environment and its dependencies are
trusted.

**What NeuroFence is for.** Recording model identity, detecting file-level
tampering, and gathering activation evidence for later analysis.

**What NeuroFence is not.** It is not a malware sandbox, not an antivirus, and
not a substitute for provenance controls on where your models come from.

---

## Safe defaults

These are the defaults; changing them is a deliberate act.

- `model.trust_remote_code: false` — enabling it executes arbitrary Python
  shipped alongside the weights. The config layer **refuses to start** unless
  `sandbox.allow_remote_code` is *also* set to true, so it cannot be turned on
  by a single careless edit.
- `model.allow_download: false` — scans never fetch. Only
  `scripts/prepare_model.py --download` touches the network, and it requires an
  explicit flag.
- `sandbox.block_sockets: true` — non-loopback connections raise
  `SandboxViolationError` and are recorded in the run manifest.
- Model loading prefers safetensors. If a model ships **only** pickle-format
  weights (`.bin`, `.pt`, `.pth`, `.ckpt`), NeuroFence logs a warning: loading
  those deserialises arbitrary objects and can execute code before any of this
  tool's logic runs.

---

## Pickle weights

`.bin` / `.pt` / `.pth` / `.ckpt` files are Python pickles. `torch.load` on an
untrusted pickle is remote code execution — it happens during loading, before
NeuroFence sees a single activation. No sandbox in this codebase prevents it.

Prefer safetensors. If you must load pickle weights from an untrusted source,
do it in a disposable VM.

---

## Handling evidence

- `models/`, `data/`, and `logs/` are gitignored. Do not commit weights or scan
  output — they may be evidence, and they are large.
- Run artefacts record file paths, model identifiers, and library versions.
  Review `manifest.json` before sharing it outside your organisation.
- The fingerprint recorded in `metadata.json` reflects the model **as loaded**.
  If you override `--model-path` without `--name`, the recorded identifier
  becomes `local:<dirname>` rather than the config's model name, so an artefact
  never claims provenance it does not have.

---

## Controlled test artefacts (Phase 2)

`scripts/run_experiment.py` and `neurofence.experiments.controlled_backdoor`
deliberately create a **modified model** to validate the detector against known
ground truth.

**What it is.** The modification scales the input-embedding row for a trigger
token, producing a larger activation signature when that token appears. It is
an activation marker: it teaches the model nothing, changes no outputs, and is
not a functional backdoor. It exists to test the measurement instrument, in the
same spirit as an EICAR antivirus test file.

**Safeguards in the code.**

- Never modifies a model in place — a separate destination is required, and
  passing the source path raises.
- Special tokens (`<unk>`, `<pad>`, BOS, EOS) are excluded from amplification.
  Amplifying `<unk>` would alter behaviour on *every* input.
- Refuses to run when the trigger's token ids also appear in ordinary text
  (character-level or aggressively-splitting tokenizers). Modifying shared rows
  contaminates the baseline and makes the experiment meaningless.
- Writes `BACKDOOR_README.txt` and `backdoor_manifest.json` into the output so a
  test artefact cannot be mistaken for a clean model.
- Verifies the weights hash actually changed, and raises if it did not.

**Your responsibilities.**

- Do not distribute or deploy a directory containing `BACKDOOR_README.txt`.
- `models/` and `data/` are gitignored; keep experiment output out of version
  control and off model registries.
- If you extend the experiment to modify behaviour rather than activations, you
  are building something materially different from what ships here. Do not do
  that on a model anyone else can reach.

---

## Interpreting scan results

A high risk score is **evidence strength for the trigger hypothesis, not
probability of compromise.**

- A positive verdict means an activation anomaly reproduced across carrier
  sentences and exceeded matched control tokens. It does not establish intent,
  a functional backdoor, or malicious origin.
- A negative verdict means *this scan, with this trigger, found nothing*. It is
  not a clean bill of health. NeuroFence tests candidate triggers you supply;
  it does not search the space of all possible triggers.
- `inconclusive_no_controls` and `measurement_unreliable` are not weak
  negatives. They mean no conclusion is available, and the score is capped or
  zeroed accordingly.

Do not report a NeuroFence score as a security certification of a third-party
model.

---

## Prompt data

Fuzzer prompts are generated, not user data, but `fuzzer.log_prompt_text`
defaults to `false` so prompt text stays out of logs and reports. If you extend
the fuzzer with prompts drawn from real traffic, keep that setting off.

---

## Reports as artefacts (Phase 3)

A PDF outlives the scan that produced it and gets forwarded to people who never
saw the console output. The report is therefore written to be safe when read in
isolation:

- The **Limitations** section is emitted on every report, including clean ones.
- Verdicts use a fixed label table. `trigger_behaviour_detected` renders as
  "Potential Backdoor Indicator", never "Confirmed Malware". The terms
  *confirmed*, *malware*, *malicious*, and *compromised* do not appear in the
  report vocabulary.
- A negative result is stated explicitly as "not a clean bill of health".
- Detection rate, false positive rate, precision, recall, and F1 appear only
  when a controlled experiment actually ran. They are never estimated.
- Charts are rendered only from measured data. A figure whose data is missing is
  omitted rather than replaced with an illustrative placeholder.

**Do not** present a NeuroFence PDF as certification that a third-party model is
safe. It records what one scan measured about one analyst-supplied trigger.

Reports may contain local filesystem paths, model identifiers, and environment
details. Review before sharing outside your organisation.

---

## Desktop application

The GUI applies the same sandbox as the CLI: no network, no downloads, no
remote code execution. It adds no privileged capability of its own.

Two behaviours worth knowing:

- Scanning happens on a worker thread. Cancellation is cooperative and takes
  effect between stages, so a cancelled scan may continue briefly rather than
  stopping instantly. The thread is always joined before the app exits.
- The Configure screen refuses to start scans that cannot produce a meaningful
  result — missing control tokens, a trigger listed as its own control, or an
  undersized baseline. These are validation failures, not warnings, because
  each one would otherwise yield a confident-looking but meaningless score.

---

## Reporting a vulnerability

Report security issues in NeuroFence itself privately to the maintainers rather
than opening a public issue. Include the version, the config used, and a
minimal reproduction.

Findings about a *third-party model* are not NeuroFence vulnerabilities — take
those to the model publisher.
