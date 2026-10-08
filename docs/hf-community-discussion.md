# Jawbreaker Project Summary

Initiative: ML Empowerment Build Challenge 3.0  
Maintainer: The Jawbreaker Team

Jawbreaker helps people pause before clicking, replying, sharing a code, or sending money. Paste a suspicious message to receive a risk assessment, a short explanation of warning signs, and a safer next step.

The local interface uses MiniCPM5-1B Q4_K_M in GGUF format through `llama-cpp-python`. The downloaded GGUF is the base model, not a fine-tuned Jawbreaker adapter. The app validates generated output and applies deterministic safety checks before presenting a result.

Evaluation assets and setup instructions are maintained in this repository:

- `SETUP.md` describes installation and local launch.
- `eval/README.md` describes datasets, backends, and evaluation commands.
- `HONEST_SUBMISSION.md` documents supported claims and limitations.
