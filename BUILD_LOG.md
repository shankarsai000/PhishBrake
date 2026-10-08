# Jawbreaker Build Log

Initiative: ML Empowerment Build Challenge 3.0  
Maintainer: The Jawbreaker Team

## Product

Jawbreaker is a focused scam-safety interface. It turns a suspicious message into a risk assessment, warning signs, and a safer next step without requiring a hosted language-model API.

## Runtime

- The default local backend is `llama-cpp-python` with the MiniCPM5-1B Q4_K_M GGUF base model.
- The optional Transformers backend can be configured separately.
- The local GGUF does not include the historical Jawbreaker LoRA adapter.

## Safety and Evaluation

- Model responses are parsed against a schema and unsafe action recommendations are repaired before display.
- A deterministic heuristic fallback handles failed or under-calling model responses.
- Synthetic and sanitized datasets, evaluation runners, and reports are maintained under `eval/` and `training/`.
- Historical MiniCPM5-1B LoRA evaluation results remain comparison evidence and do not describe the default GGUF runtime.
