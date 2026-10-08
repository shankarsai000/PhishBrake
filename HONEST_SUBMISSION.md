# Project Claims and Limitations

Initiative: ML Empowerment Build Challenge 3.0  
Maintainer: The Jawbreaker Team

## Supported Claims

- Jawbreaker helps users pause and verify suspicious messages before acting.
- The default local runtime uses MiniCPM5-1B in GGUF format through `llama-cpp-python`.
- The default GGUF is the base model; it does not include the separately trained Jawbreaker LoRA adapter.
- The app validates structured output, checks unsafe action recommendations, and falls back to deterministic analysis when needed.
- Evaluation and generated training examples are synthetic or sanitized and are stored in the repository.
- Historical MiniCPM LoRA reports describe those evaluated configurations only; they do not describe the default local GGUF.

## Boundaries

- Jawbreaker is not legal, financial, or cybersecurity advice.
- Do not claim measured real-user outcomes without consent and documented feedback.
- Do not describe synthetic data as private or real-user conversation data.
- Do not claim a model evaluation score for a different model, quantization, or runtime.
- Users should verify through official apps, websites, or known contact information rather than links or numbers in suspicious messages.
