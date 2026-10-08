from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "jawbreaker-minicpm-eval"
REMOTE_ROOT = Path("/workspace")
REMOTE_OUTPUT = Path("/outputs")
LOCAL_ROOT = Path(__file__).resolve().parents[1]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==1.12.0",
        "huggingface-hub>=0.34.0,<1.0",
        "peft>=0.18.0,<1.0",
        "sentencepiece>=0.2.0,<1.0",
        "torch==2.9.1",
        "transformers==4.57.3",
    )
    .add_local_dir(LOCAL_ROOT / "jawbreaker", remote_path=REMOTE_ROOT / "jawbreaker")
    .add_local_dir(LOCAL_ROOT / "eval", remote_path=REMOTE_ROOT / "eval")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("jawbreaker-training", create_if_missing=True)


@app.function(
    gpu="A100",
    timeout=3 * 60 * 60,
    volumes={REMOTE_OUTPUT: volume},
)
def run_eval(
    dataset: str = "eval/generated_eval.jsonl",
    model_id: str = "openbmb/MiniCPM4.1-8B",
    adapter_id: str | None = "build-small-hackathon/jawbreaker-minicpm-lora",
    limit: int | None = None,
    output_prefix: str = "jawbreaker-minicpm-lora-generated",
    apply_safety_guard: bool = False,
) -> None:
    import os
    import subprocess

    os.chdir(REMOTE_ROOT)
    report_path = REMOTE_OUTPUT / f"{output_prefix}.json"
    predictions_path = REMOTE_OUTPUT / f"{output_prefix}.predictions.jsonl"
    cmd = [
        "python",
        "eval/run_eval.py",
        "--backend",
        "transformers",
        "--model-id",
        model_id,
        "--dataset",
        dataset,
        "--attn-implementation",
        "eager",
        "--predictions-out",
        str(predictions_path),
        "--json-out",
        str(report_path),
    ]
    if adapter_id:
        cmd.extend(["--adapter-id", adapter_id])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if apply_safety_guard:
        cmd.append("--apply-safety-guard")

    print("jawbreaker modal_eval command=" + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"jawbreaker modal_eval report={report_path}", flush=True)
    print(f"jawbreaker modal_eval predictions={predictions_path}", flush=True)


@app.local_entrypoint()
def main(
    dataset: str = "eval/generated_eval.jsonl",
    model_id: str = "openbmb/MiniCPM4.1-8B",
    adapter_id: str | None = "build-small-hackathon/jawbreaker-minicpm-lora",
    limit: int | None = None,
    output_prefix: str = "jawbreaker-minicpm-lora-generated",
    apply_safety_guard: bool = False,
) -> None:
    run_eval.remote(
        dataset=dataset,
        model_id=model_id,
        adapter_id=adapter_id,
        limit=limit,
        output_prefix=output_prefix,
        apply_safety_guard=apply_safety_guard,
    )
