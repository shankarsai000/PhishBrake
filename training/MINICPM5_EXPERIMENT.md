# MiniCPM5-1B Evaluation Experiment

This experiment became the production path.

The shipped Jawbreaker app now uses `openbmb/MiniCPM5-1B` with
`build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8`.

## Why Try It

`openbmb/MiniCPM5-1B` is a compact model suited to Jawbreaker's narrow
scam-defense task.

## Decision Rule

Promote the 1B path only if it has:

- `0` dangerous-as-safe errors on `eval/hard_v3_eval.jsonl`
- `0` dangerous-as-needs-check errors on `eval/hard_v3_eval.jsonl`
- `0` unsafe action violations
- `0` invalid predictions after schema repair
- acceptable local inference latency
- no obvious regression on legitimate/safe examples

The v8 adapter cleared this bar on the completed 632-case hard guarded eval:

- `632` cases
- `91.61%` risk accuracy
- `88.77%` scam type accuracy
- `90.69%` mean tactic recall
- `0` dangerous-as-safe
- `0` dangerous-as-needs-check
- `0` safe-as-dangerous-or-suspicious
- `0` unsafe action violations
- `0` invalid predictions
- `0` model errors

## Step 1: Base Model Smoke Eval

Run a 20-case smoke test before training:

```bash
modal run training/modal_eval.py \
  --model-id openbmb/MiniCPM5-1B \
  --adapter-id '' \
  --dataset eval/hard_v3_eval.jsonl \
  --limit 20 \
  --output-prefix jawbreaker-minicpm5-1b-base-smoke20
```

Continue only if the model loads and produces mostly valid predictions.

## Step 2: Train LoRA

Use the same v3 contrastive training data as the shipped 8B adapter:

```bash
modal run training/modal_train.py \
  --model-id openbmb/MiniCPM5-1B \
  --train-file training/data/train_v3.jsonl \
  --dev-file training/data/dev_v3.jsonl \
  --output-name jawbreaker-minicpm5-1b-lora-v1 \
  --epochs 3 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --lr-scheduler-type cosine \
  --max-length 768 \
  --batch-size 2 \
  --grad-accum 8 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --push-to-hub \
  --hub-model-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v1
```

The larger per-device batch is only a starting point. If Modal logs show memory
pressure, use `--batch-size 1 --grad-accum 16`.

## Step 3: Hard Eval

Run the full 215-case hard eval:

```bash
modal run training/modal_eval.py \
  --model-id openbmb/MiniCPM5-1B \
  --adapter-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v1 \
  --dataset eval/hard_v3_eval.jsonl \
  --output-prefix jawbreaker-minicpm5-1b-lora-v1-hard215
```

Run the 100-case product eval:

```bash
modal run training/modal_eval.py \
  --model-id openbmb/MiniCPM5-1B \
  --adapter-id build-small-hackathon/jawbreaker-minicpm5-1b-lora-v1 \
  --dataset eval/scam_eval.jsonl \
  --output-prefix jawbreaker-minicpm5-1b-lora-v1-scam100
```

## Step 4: Promotion

Only if both evals clear the decision rule:

1. Add reports under `eval/reports/`.
2. Update README with the verified compact-model details.
3. Set the deployment variables:
   - `JAWBREAKER_TRANSFORMERS_MODEL_ID=openbmb/MiniCPM5-1B`
   - `JAWBREAKER_ADAPTER_ID=build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8`
4. Smoke test desktop and mobile.

The completed hard632 v8 eval is committed under `eval/reports/` as final evidence. The earlier hard394 and hard320 v4 reports remain comparison evidence.
