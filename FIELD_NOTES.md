# Field Notes

Initiative: ML Empowerment Build Challenge 3.0  
Maintainer: The Jawbreaker Team

These notes capture product and engineering decisions, not user research or measured outcomes.

## Product Scope

Jawbreaker is a pause-and-verify tool for suspicious texts, emails, and direct messages. The interface presents a risk level, warning signs, and one safe next step, with a note users can share with someone they trust.

## Runtime

- The default local backend is `llama-cpp-python` loading MiniCPM5-1B Q4_K_M GGUF.
- The GGUF contains the base model only; the separately trained LoRA adapter is not merged or attached.
- Transformers remains an optional backend for evaluating or deploying other configured model variants.
- The app does not require a hosted LLM API for local inference.

## Safety Design

- Validate model output against the application schema.
- Replace unsafe actions such as following suspicious links or submitting credentials.
- Use deterministic heuristic analysis if inference fails or misses a clear danger signal.
- Encourage verification through an official app, website, or previously trusted contact.

## Evaluation

The repository contains synthetic and sanitized datasets, generators, regression tests, and model reports. Historical MiniCPM5-1B LoRA v8 reports cover 632 cases; their scores apply only to that adapter configuration, not the default base GGUF. See `eval/README.md` and `DEVELOPMENT_EVIDENCE.md`.

## Privacy

Do not add raw private chats, contact details, account information, or identifying user stories to public datasets or documentation. Synthetic and sanitized examples are used for regression and evaluation.
