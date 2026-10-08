# Jawbreaker Run Context

## Project Identity

- Project: Jawbreaker
- Initiative: ML Empowerment Build Challenge 3.0
- Maintainer: The Jawbreaker Team
- Purpose: help someone pause before clicking a suspicious link, replying, sharing a code, calling a number, or sending money.
- Audience: people who want a calm, plain-language safety check, including family members helping someone else verify a message.

Jawbreaker is not a general chatbot, security scanner, or replacement for official support. It turns a suspicious text, email, or DM into a risk assessment, evidence, and one safer next step. It is not legal, financial, or cybersecurity advice.

## Workspace Paths

This checkout has a nested project directory:

- Project root: `C:\jawbreaker-main\jawbreaker-main`
- Virtual environment: `C:\jawbreaker-main\venv`
- Default GGUF: `C:\jawbreaker-main\jawbreaker-main\models\minicpm5-1b-gguf\minicpm5-1b-Q4_K_M.gguf`

The project root is the directory containing `app.py`, `jawbreaker/`, `tests/`, `eval/`, and `training/`. Running `python app.py` from `C:\jawbreaker-main` fails because there is no `C:\jawbreaker-main\app.py`.

## Run the App

From the parent folder `C:\jawbreaker-main`:

```powershell
Set-Location .\jawbreaker-main
..\venv\Scripts\python.exe app.py
```

From the project root `C:\jawbreaker-main\jawbreaker-main`:

```powershell
..\venv\Scripts\python.exe app.py
```

The Gradio server prints its local URL, normally `http://127.0.0.1:7860`. Stop it with `Ctrl+C` in the terminal running the process.

## Runtime and Model

- Default backend: `llama-cpp-python` (`JAWBREAKER_BACKEND=llama-cpp`).
- Installed/pinned engine version: `llama-cpp-python==0.3.36`.
- Default model: MiniCPM5-1B Q4_K_M GGUF, downloaded from `Abiray/MiniCPM5-1B-GGUF`.
- Default path: `models/minicpm5-1b-gguf/minicpm5-1b-Q4_K_M.gguf`, relative to the project root.
- The GGUF is the base MiniCPM5-1B model. It does not contain or attach the Jawbreaker LoRA v8 adapter.
- A Transformers backend is also supported. It can load `openbmb/MiniCPM5-1B` and uses the configured/default adapter ID when selected. Historical adapter scores refer to that evaluated adapter configuration, not the default GGUF.
- The legacy `build-small-hackathon/...` adapter identifier remains in optional adapter/evaluation paths because it is the exact Hugging Face model ID. It is a model locator, not project ownership or attribution.

Useful environment variables:

- `JAWBREAKER_BACKEND`: `llama-cpp` by default; can be set to `transformers`, `zerogpu`, or `heuristic` where supported.
- `JAWBREAKER_MODEL_PATH`: override the local GGUF path.
- `JAWBREAKER_MODEL_REPO` and `JAWBREAKER_MODEL_FILE`: override the Hugging Face source used when the local path is missing.
- `JAWBREAKER_N_CTX`, `JAWBREAKER_N_THREADS`, `JAWBREAKER_N_GPU_LAYERS`: llama.cpp context, CPU threads, and GPU offload controls.
- `JAWBREAKER_TRANSFORMERS_MODEL_ID` and `JAWBREAKER_ADAPTER_ID`: configure the optional Transformers model and adapter.

## Analysis and Safety Flow

1. The Gradio UI accepts a message and session-local history.
2. Very short or low-context messages receive a cautious `needs_check` response.
3. Otherwise, the selected analyzer runs: llama.cpp by default, with Transformers and heuristic alternatives.
4. Model output is parsed as JSON and checked against the expected schema.
5. `repair_prediction()` normalizes malformed fields and replaces known unsafe recommendations, including advice to submit passwords or credentials.
6. The deterministic heuristic guard can replace a model under-call with heuristic analysis; inference failures also fall back to heuristics.
7. The UI renders the verdict, warning signs, safe action, and a copyable trusted-person note.

The app should direct users to verify through an official app, a known website, or a trusted contact, not through a link or number in a suspicious message. Avoid putting real passwords, account details, or private personal data into sample messages.

## Verify the Model and Tests

Run the GGUF load check from the project root:

```powershell
..\venv\Scripts\python.exe -c "from llama_cpp import Llama; from app import resolve_model_path; Llama(model_path=str(resolve_model_path()), n_ctx=2048, verbose=False); print('Model loaded successfully!')"
```

Run the tests:

```powershell
..\venv\Scripts\python.exe -m pytest tests -q
```

At the last verification, all 45 tests passed. The tests cover schema handling, safety guards, trust notes, evaluation data, and the app path. Re-run them after changes.

## Repository Map

- `app.py`: Gradio/FastAPI entry point, embedded UI, backend selection, model resolution, analysis orchestration, and status labels.
- `style.css`: custom visual design for the Gradio interface.
- `jawbreaker/contract.py`: model system prompt and JSON response contract.
- `jawbreaker/analyzers.py`: llama.cpp and Transformers adapters, output parsing, schema repair, and unsafe-action checks.
- `jawbreaker/schema.py`: `ScamAnalysis` representation and heuristic analysis.
- `jawbreaker/trust.py`: low-context handling, confidence metadata, and shareable safety notes.
- `jawbreaker/render.py`: result and memory rendering helpers.
- `tests/`: focused regression tests; run with pytest.
- `eval/`: synthetic/sanitized datasets, runner, saved predictions, and historical reports.
- `training/`: synthetic data generators and optional PEFT/LoRA and Modal workflows.
- `models/`: local model files; large weights are ignored by Git.
- `SETUP.md`: short local setup and launch instructions.
- `README.md`: project overview, limitations, model and evaluation notes.
- `FIELD_NOTES.md`, `BUILD_LOG.md`, `DEVELOPMENT_TRACE.md`, `DEVELOPMENT_EVIDENCE.md`: team-maintained project notes and evidence.

## Evaluation Notes

The repository contains synthetic and sanitized evaluation examples. The historical MiniCPM5-1B LoRA v8 report covers 632 cases and records 579/632 risk accuracy (91.61%), zero dangerous-as-safe cases, and zero unsafe-action violations. Those metrics are specific to the reported LoRA configuration; they must not be presented as a score for the default base GGUF without separately evaluating that exact GGUF/runtime combination.


cd .\jawbreaker-main
..\venv\Scripts\python.exe app.py
