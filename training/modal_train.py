from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "jawbreaker-minicpm-lora"
REMOTE_ROOT = Path("/workspace")
REMOTE_OUTPUT = Path("/outputs")
LOCAL_ROOT = Path(__file__).resolve().parents[1]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==1.12.0",
        "datasets>=4.4.0,<5.0",
        "huggingface-hub>=0.34.0,<1.0",
        "peft>=0.18.0,<1.0",
        "sentencepiece>=0.2.0,<1.0",
        "torch==2.9.1",
        "transformers==4.57.3",
    )
    .add_local_dir(LOCAL_ROOT / "jawbreaker", remote_path=REMOTE_ROOT / "jawbreaker")
    .add_local_dir(LOCAL_ROOT / "training", remote_path=REMOTE_ROOT / "training")
    .add_local_dir(LOCAL_ROOT / "eval", remote_path=REMOTE_ROOT / "eval")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("jawbreaker-training", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=6 * 60 * 60,
    volumes={REMOTE_OUTPUT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def train_lora(
    model_id: str = "openbmb/MiniCPM4.1-8B",
    output_name: str = "jawbreaker-minicpm-lora",
    epochs: float = 1.0,
    train_file: str = "training/data/train.jsonl",
    dev_file: str = "training/data/dev.jsonl",
    max_length: int = 768,
    batch_size: int = 1,
    grad_accum: int = 16,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.0,
    lr_scheduler_type: str = "linear",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
) -> None:
    import os
    import subprocess

    os.chdir(REMOTE_ROOT)
    output_dir = REMOTE_OUTPUT / output_name
    cmd = [
        "python",
        "training/train_lora.py",
        "--model-id",
        model_id,
        "--train-file",
        train_file,
        "--dev-file",
        dev_file,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--max-length",
        str(max_length),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--learning-rate",
        str(learning_rate),
        "--warmup-ratio",
        str(warmup_ratio),
        "--weight-decay",
        str(weight_decay),
        "--lr-scheduler-type",
        lr_scheduler_type,
        "--lora-r",
        str(lora_r),
        "--lora-alpha",
        str(lora_alpha),
        "--lora-dropout",
        str(lora_dropout),
    ]
    if push_to_hub:
        cmd.append("--push-to-hub")
        if hub_model_id:
            cmd.extend(["--hub-model-id", hub_model_id])

    subprocess.run(cmd, check=True)
    volume.commit()


@app.local_entrypoint()
def main(
    model_id: str = "openbmb/MiniCPM4.1-8B",
    output_name: str = "jawbreaker-minicpm-lora",
    epochs: float = 1.0,
    train_file: str = "training/data/train.jsonl",
    dev_file: str = "training/data/dev.jsonl",
    max_length: int = 768,
    batch_size: int = 1,
    grad_accum: int = 16,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.0,
    lr_scheduler_type: str = "linear",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
) -> None:
    train_lora.remote(
        model_id=model_id,
        output_name=output_name,
        epochs=epochs,
        train_file=train_file,
        dev_file=dev_file,
        max_length=max_length,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )
