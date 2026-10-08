# Jawbreaker Eval Set

`scam_eval.jsonl` is the project compass for model selection.

The first version contains 100 synthetic/sanitized examples across:

- dangerous scams
- suspicious messages
- legitimate messages that still need verification
- safe benign messages

Primary metrics:

- valid structured output
- exact risk-level match
- dangerous scams mislabeled as safe
- safe messages mislabeled as dangerous or suspicious
- action safety
- tactic recall
- latency

The eval intentionally includes legitimate alerts and ordinary messages. A scam detector that calls everything dangerous is not useful for the person Jawbreaker is built to protect.

There is also a generated holdout set:

```bash
python3 training/generate_jawbreaker_data.py
python3 eval/run_eval.py --dataset eval/generated_eval.jsonl --backend heuristic
```

The generated set is for scale and regression pressure. The 100-case hand-curated set remains the product compass.

## Fresh 2026 Pattern Eval

`fresh_2026_scam_eval.jsonl` is a separate held-out eval enrichment set, not training data.

It contains 100 synthetic/sanitized cases modeled after current public scam patterns:

- fake toll / parking / DMV smishing
- package delivery phishing
- bank, PayPal, and crypto callback phishing
- WhatsApp / Telegram task-job scams
- wrong-number crypto investment grooming
- MFA / verification-code theft
- government, tax, benefit, and Medicare impersonation
- marketplace overpayment and off-platform payment scams
- tech-support and remote-access scams
- legitimate-but-verify notices and safe benign messages

Risk mix:

- 72 dangerous
- 16 needs_check
- 12 safe

Use this set to strengthen the eval story after the model is already selected:

```bash
python3 eval/run_eval.py \
  --backend transformers \
  --model-id openbmb/MiniCPM5-1B \
  --adapter-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8 \
  --trust-remote-code \
  --attn-implementation eager \
  --apply-safety-guard \
  --dataset eval/fresh_2026_scam_eval.jsonl \
  --predictions-out eval/predictions/jawbreaker-minicpm5-1b-lora-v8-fresh2026.predictions.jsonl \
  --json-out eval/reports/jawbreaker-minicpm5-1b-lora-v8-fresh2026.json
```

Modal:

```bash
modal run training/modal_eval.py \
  --dataset eval/fresh_2026_scam_eval.jsonl \
  --model-id openbmb/MiniCPM5-1B \
  --adapter-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8 \
  --output-prefix jawbreaker-minicpm5-1b-lora-v8-fresh2026 \
  --apply-safety-guard
```

The first v4 run on this set found no dangerous undercalls and no invalid JSON, but it did expose calibration gaps that later v7/v8 work targeted:

- wrong-number crypto and marketplace scams were sometimes marked `suspicious` instead of `dangerous`
- a few safe family/school messages were over-called

Those findings feed the separate v7 calibration generator. The fresh eval rows themselves remain held out.

## v7 Calibration Eval

`hard_v7_eval.jsonl` is generated from `training/generate_v7_data.py`. It expands the hard eval with fresh-pattern calibration cases while preserving older anchors:

- wrong-number crypto grooming
- marketplace overpayment, courier-fee, and code-theft scams
- task-job and prepaid workbench scams
- MFA / verification-code theft
- toll, tax, parking, benefit, and government impersonation
- safe family/school/clinic hard negatives
- official-route `needs_check` notices

Generate it with:

```bash
python3 training/generate_v7_data.py
```

Modal eval command for a candidate v7 adapter:

```bash
modal run training/modal_eval.py \
  --dataset eval/hard_v7_eval.jsonl \
  --model-id openbmb/MiniCPM5-1B \
  --adapter-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v7 \
  --output-prefix jawbreaker-minicpm5-1b-lora-v7-hard558-guarded \
  --apply-safety-guard
```

## v8 Failure-Driven Eval

`hard_v8_eval.jsonl` is generated from `training/generate_v8_data.py`. It extends v7 with a narrow failure-driven calibration set:

- wrong-number crypto / gold / trading grooming labeled `dangerous`
- wrong-number social messages without money or investment asks labeled `suspicious`
- normal family dinner, school pickup, pharmacy, and clinic logistics labeled `safe`
- school, clinic, and pharmacy payment-link variants labeled `dangerous`
- official-route service notices labeled `needs_check`

Generate it with:

```bash
python3 training/generate_v8_data.py
```

Modal eval command for a candidate v8 adapter:

```bash
modal run training/modal_eval.py \
  --dataset eval/hard_v8_eval.jsonl \
  --model-id openbmb/MiniCPM5-1B \
  --adapter-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8 \
  --output-prefix jawbreaker-minicpm5-1b-lora-v8-hard632-guarded \
  --apply-safety-guard
```

Promotion starts with the fresh held-out eval, not the generated v8 eval:

```bash
modal run training/modal_eval.py \
  --dataset eval/fresh_2026_scam_eval.jsonl \
  --model-id openbmb/MiniCPM5-1B \
  --adapter-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8 \
  --output-prefix jawbreaker-minicpm5-1b-lora-v8-fresh2026-guarded \
  --apply-safety-guard
```

## Current Runtime Decision

The deployed Space uses `openbmb/MiniCPM5-1B` through Transformers on ZeroGPU with the published adapter `build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8`.

The GGUF / `llama-cpp-python` path is the default local runtime. The app also uses a deterministic heuristic guard so an obvious high-risk scam is not rendered as safe if the small model under-calls the risk.

`Qwen/Qwen3-0.6B` remains available as an optional Transformers evaluation candidate.

## Current Results

MiniCPM5-1B LoRA v8:

- Adapter: `build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8`
- 632-case hard guarded eval: `579/632` risk accuracy (`91.61%`), `0` dangerous-as-safe, `0` dangerous-as-needs-check, `0` safe-as-dangerous-or-suspicious, `0` unsafe action violations, `0` invalid predictions, `0` model errors

Earlier comparison evidence, MiniCPM5-1B LoRA v4:

- Adapter: `build-small-hackathon/jawbreaker-minicpm5-1b-lora-v4`
- 394-case hard guarded eval: `379/394` risk accuracy (`96.19%`), `0` dangerous-as-safe, `0` dangerous-as-needs-check, `0` suspicious-as-safe, `0` unsafe action violations, `0` invalid predictions, `0` model errors
- 320-case hard guarded eval: `310/320` risk accuracy (`96.88%`), `0` dangerous-as-safe, `0` dangerous-as-needs-check, `0` suspicious-as-safe, `0` unsafe action violations, `0` invalid predictions, `0` model errors

## Running Backends

Heuristic baseline:

```bash
python3 eval/run_eval.py --backend heuristic
```

OpenBMB MiniCPM through Transformers:

```bash
python3 eval/run_eval.py \
  --backend transformers \
  --model-id openbmb/MiniCPM5-1B \
  --trust-remote-code \
  --dataset eval/generated_eval.jsonl
```

Published final v8 adapter through Transformers:

```bash
python3 eval/run_eval.py \
  --backend transformers \
  --model-id openbmb/MiniCPM5-1B \
  --adapter-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8 \
  --trust-remote-code \
  --attn-implementation eager \
  --dataset eval/hard_v8_eval.jsonl \
  --apply-safety-guard
```

Saved prediction replay:

```bash
python3 eval/run_eval.py --backend predictions --predictions eval/predictions/model.jsonl
```

Local GGUF model through `llama-cpp-python`:

```bash
python3 eval/run_eval.py \
  --backend llama-cpp \
  --model-path models/model.gguf \
  --predictions-out eval/predictions/model.jsonl \
  --json-out eval/reports/model.json
```
