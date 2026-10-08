# Development Evidence

Initiative: ML Empowerment Build Challenge 3.0  
Maintainer: The Jawbreaker Team

## Code Map

- `app.py`: Gradio interface, backend selection, and model-path resolution.
- `jawbreaker/analyzers.py`: model output parsing, schema repair, and unsafe-action checks.
- `jawbreaker/schema.py` and `jawbreaker/trust.py`: analysis structure, heuristic fallback, and shareable notes.
- `eval/run_eval.py`: evaluation runner for heuristic, Transformers, llama.cpp, and saved-prediction backends.
- `training/`: synthetic data generation and optional PEFT/LoRA workflows.

## Historical Model Evaluation

The archived MiniCPM5-1B LoRA v8 report covers 632 cases and records 579/632 risk accuracy (91.61%), zero dangerous-as-safe cases, zero dangerous-as-needs-check cases, zero unsafe actions, zero invalid predictions, and zero model errors. See `eval/reports/jawbreaker-minicpm5-1b-lora-v8-hard632-safetyguard-v4.json`.

These metrics describe that evaluated LoRA configuration, not the default local base GGUF. The local model path is tested independently with the installed llama.cpp backend.
