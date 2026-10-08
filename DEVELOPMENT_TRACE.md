# Development Trace

This document records major Jawbreaker engineering decisions. Maintainer: The Jawbreaker Team.

## Product Direction

- Focused the app on helping people pause before clicking, replying, sharing a code, or sending money.
- Chose a safety-card interface with a clear verdict, evidence, and trusted-person handoff.
- Added schema validation, deterministic action checks, and a heuristic fallback around model output.

## Model and Evaluation

- Built synthetic and sanitized evaluation sets for dangerous, suspicious, needs-check, and safe messages.
- Evaluated MiniCPM model and LoRA candidates with the same guarded runner.
- Kept historical adapter reports as comparison artifacts while distinguishing them from the default local base GGUF.
- Added a local llama.cpp runtime using MiniCPM5-1B Q4_K_M GGUF.

## Initiative

Jawbreaker is maintained by The Jawbreaker Team for ML Empowerment Build Challenge 3.0. Public documentation omits personal names, private messages, contact details, and individual social profiles.
