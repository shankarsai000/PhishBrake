# Jawbreaker Training

This folder contains the OpenBMB MiniCPM fine-tuning path for Jawbreaker.

The goal is not to make a general scam chatbot. The target behavior is narrow:

1. Read one suspicious message.
2. Return valid JSON matching Jawbreaker's schema.
3. Avoid dangerous false negatives.
4. Avoid calling normal messages dangerous.
5. Recommend a safe verification path.

## Generate Data

The generated data is synthetic and deterministic. It avoids private messages and keeps links on `.example` domains.

```bash
python3 training/generate_jawbreaker_data.py
```

Default outputs:

- `training/data/train.jsonl`
- `training/data/dev.jsonl`
- `training/data/test.jsonl`
- `eval/generated_eval.jsonl`

## Evaluate Baseline MiniCPM

```bash
python3 eval/run_eval.py \
  --backend transformers \
  --model-id openbmb/MiniCPM4.1-8B \
  --trust-remote-code \
  --dataset eval/generated_eval.jsonl \
  --predictions-out eval/predictions/minicpm4_1_8b_generated.jsonl \
  --json-out eval/reports/minicpm4_1_8b_generated.json
```

For quick smoke tests:

```bash
python3 eval/run_eval.py \
  --backend transformers \
  --model-id openbmb/MiniCPM4.1-8B \
  --trust-remote-code \
  --dataset eval/generated_eval.jsonl \
  --limit 10
```

## Fine-Tune LoRA

Install training-only dependencies outside the Space runtime:

```bash
pip install -r requirements-train.txt
```

Run a short LoRA training pass:

```bash
python3 training/train_lora.py \
  --model-id openbmb/MiniCPM4.1-8B \
  --train-file training/data/train.jsonl \
  --dev-file training/data/dev.jsonl \
  --output-dir training/output/jawbreaker-minicpm-lora
```

## Fine-Tune on Modal

Modal is the preferred training path because MiniCPM4.1-8B should be trained on GPU hardware, not inside a hosted app runtime.

One-time local setup:

```bash
pip install modal
modal setup
```

Create a Modal secret named `huggingface-secret` with an `HF_TOKEN` value that can read the base model and push to your Hugging Face account:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
```

Run training on Modal:

```bash
modal run training/modal_train.py
```

The current shipped adapter was trained with the v3 contrastive data:

```bash
python3 training/generate_v3_data.py

modal run training/modal_train.py \
  --train-file training/data/train_v3.jsonl \
  --dev-file training/data/dev_v3.jsonl \
  --output-name jawbreaker-minicpm-lora-v3 \
  --epochs 3 \
  --learning-rate 7e-5 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --lr-scheduler-type cosine \
  --max-length 768 \
  --batch-size 1 \
  --grad-accum 16 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --push-to-hub \
  --hub-model-id build-small-hackathon/jawbreaker-minicpm-lora-v3
```

Publish a new adapter only after evaluation demonstrates an improvement:

```bash
modal run training/modal_train.py \
  --push-to-hub \
  --hub-model-id build-small-hackathon/jawbreaker-minicpm-lora
```

The Modal job writes checkpoints to the `jawbreaker-training` Modal volume under `/outputs`.

To publish an adapter after evaluation:

```bash
python3 training/train_lora.py \
  --model-id openbmb/MiniCPM4.1-8B \
  --output-dir training/output/jawbreaker-minicpm-lora \
  --push-to-hub \
  --hub-model-id build-small-hackathon/jawbreaker-minicpm-lora
```

## Deployment Decision Rule

Use the fine-tuned adapter only if it beats the base model on:

- valid JSON rate
- zero dangerous-as-safe misses
- lower false alarms on safe messages
- safe action compliance
- acceptable Space latency

If the adapter improves JSON but hurts safety, do not deploy it.

Historical candidate: `build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8`
on `openbmb/MiniCPM5-1B`. The local default uses the base model as GGUF.

- 632-case hard guarded eval: `579/632` risk accuracy (`91.61%`), no dangerous undercalls, no safe-as-dangerous-or-suspicious misses, no unsafe action violations, no invalid predictions, no model errors.
- Earlier 1B v4 evals remain useful as comparison evidence: 394-case hard guarded eval at `379/394` risk accuracy (`96.19%`) and 320-case hard guarded eval at `310/320` risk accuracy (`96.88%`), both with no dangerous undercalls.
- Earlier 8B v3 evals remain useful as comparison evidence, but the 1B v8 adapter is the final deployed model.

## v7 Calibration Experiment

`training/generate_v7_data.py` is a candidate follow-up, not the production adapter yet. It uses sanitized public-pattern examples and hand-written hard negatives to address the fresh 2026 eval gaps without training on the fresh held-out eval rows.

Generate the v7 data:

```bash
python3 training/generate_v7_data.py
```

Current generated sizes:

- `training/data/train_v7.jsonl`: 2,192 SFT rows
- `training/data/dev_v7.jsonl`: 498 SFT rows
- `eval/hard_v7_eval.jsonl`: 558 held-out hard cases

Train the candidate adapter on Modal:

```bash
modal run training/modal_train.py \
  --model-id openbmb/MiniCPM5-1B \
  --train-file training/data/train_v7.jsonl \
  --dev-file training/data/dev_v7.jsonl \
  --output-name jawbreaker-minicpm5-1b-lora-v7 \
  --epochs 2 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --lr-scheduler-type cosine \
  --max-length 768 \
  --batch-size 1 \
  --grad-accum 16 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --push-to-hub \
  --hub-model-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v7
```

Do not replace v4 unless v7 improves fresh-pattern accuracy without introducing dangerous undercalls, invalid JSON, or a higher safe false-positive rate.

## v8 Failure-Driven Calibration Experiment

`training/generate_v8_data.py` is a narrower follow-up to v7. It targets the two v7 fresh-eval failure modes:

- wrong-number crypto / gold / trading grooming was sometimes softened to `needs_check`
- ordinary family, school, and pharmacy logistics were sometimes over-called

The v8 data keeps the older hard anchors, adds contrastive wrong-number examples, and does not train on exact `fresh_2026_scam_eval.jsonl` rows.

Generate the v8 data:

```bash
python3 training/generate_v8_data.py
```

Current generated sizes:

- `training/data/train_v8.jsonl`: 2,488 SFT rows
- `training/data/dev_v8.jsonl`: 572 SFT rows
- `eval/hard_v8_eval.jsonl`: 632 held-out hard cases

Train the candidate adapter on Modal:

```bash
modal run training/modal_train.py \
  --model-id openbmb/MiniCPM5-1B \
  --train-file training/data/train_v8.jsonl \
  --dev-file training/data/dev_v8.jsonl \
  --output-name jawbreaker-minicpm5-1b-lora-v8 \
  --epochs 1.5 \
  --learning-rate 4e-5 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --lr-scheduler-type cosine \
  --max-length 768 \
  --batch-size 1 \
  --grad-accum 16 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --push-to-hub \
  --hub-model-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8
```

Promotion rule:

1. Fresh 2026 held-out eval must have `dangerous_as_safe=0`, `dangerous_as_needs_check=0`, `invalid_predictions=0`, and `unsafe_action_violations=0`.
2. Hard v8 eval must keep zero dangerous undercalls and valid JSON.
3. Safe false positives must not materially worsen versus v4/v7.

Promotion result:

- Fresh 2026 guarded eval after guard calibration: `92/100` risk accuracy, `0` dangerous-as-safe, `0` dangerous-as-needs-check, `0` safe-as-dangerous-or-suspicious, `0` unsafe action violations, `0` invalid predictions, `0` model errors.
- Hard v8 guarded eval after guard calibration: `579/632` risk accuracy (`91.61%`), `0` dangerous-as-safe, `0` dangerous-as-needs-check, `0` safe-as-dangerous-or-suspicious, `0` unsafe action violations, `0` invalid predictions, `0` model errors.
