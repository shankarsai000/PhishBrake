# Jawbreaker: Scam Safety Before You Act

Initiative: ML Empowerment Build Challenge 3.0  
Maintainer: The Jawbreaker Team

Scam messages are designed to create urgency. A package is held, a bank account is at risk, or a family member urgently needs money. Jawbreaker gives people a moment to pause and verify before taking action.

## What It Does

Paste a suspicious text, email, or direct message. Jawbreaker returns a risk level, identifies the sender's apparent claim and pressure tactics, describes what they are asking for, and recommends a safer next step. A shareable note can help involve someone the user trusts.

The app is intentionally narrow. It is not a general chatbot and does not replace official support channels or human judgment.

## Local Model

The default local runtime uses MiniCPM5-1B Q4_K_M in GGUF format through `llama-cpp-python`. This is the base model. It does not include the separately trained Jawbreaker LoRA adapter. Transformers and adapter evaluation remain optional paths.

The local runtime does not require a hosted language-model API. Model output is parsed and checked before display; deterministic heuristics provide a fallback if inference fails or underestimates a clear risk.

## Evaluation and Limits

The repository includes synthetic and sanitized training/evaluation examples, regression tests, and reports. Historical adapter results apply only to the specific model configuration evaluated and should not be attributed to the default GGUF.

Jawbreaker is a safety aid, not legal, financial, or cybersecurity advice. Never follow a link or call a number solely because a suspicious message instructs you to. Verify through an official app, a known website, or a contact method you already trust.

## Run Locally

See [`SETUP.md`](../SETUP.md) for installation and startup steps. The Gradio interface runs locally at `http://127.0.0.1:7860`.
