# Project Overview

Initiative: ML Empowerment Build Challenge 3.0  
Maintainer: The Jawbreaker Team

Jawbreaker is a local-first scam-safety application. It accepts a suspicious message and returns a risk assessment, warning signs, and a safer next step.

## Runtime

- Default backend: `llama-cpp-python` with MiniCPM5-1B Q4_K_M GGUF.
- Optional backend: Transformers, configured independently.
- The default GGUF is the base model and does not contain the historical Jawbreaker LoRA adapter.
- Gradio serves the local interface at `http://127.0.0.1:7860`.

## Safety and Evaluation

- Structured model output is validated and unsafe action advice is repaired before display.
- Heuristic analysis provides a fallback when model inference fails or under-calls a clear danger signal.
- Synthetic and sanitized examples are kept under `eval/` and `training/data/`.
- Historical adapter results are documented separately from the default local model.

## Local Verification

```powershell
..\venv\Scripts\python.exe -m pytest tests
..\venv\Scripts\python.exe app.py
```
