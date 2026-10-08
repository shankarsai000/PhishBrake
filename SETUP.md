# Setup

## Local Project

Open a terminal at the repository root:

```bash
cd C:\jawbreaker-main\jawbreaker-main
```

Check the project files:

```bash
Get-ChildItem
```

## Install and Run Locally

Use the existing virtual environment or create one, then install the application dependencies:

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The default backend loads MiniCPM5-1B Q4_K_M from `models/minicpm5-1b-gguf/`. To use another GGUF, set `JAWBREAKER_MODEL_PATH` before starting the app.

```powershell
..\venv\Scripts\python.exe app.py
```

The interface is served locally at `http://127.0.0.1:7860`.

The app also exposes `POST /api/scan` for local integrations such as a Chrome extension:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:7860/api/scan `
  -Method Post -ContentType "application/json" `
  -Body '{"text":"Urgent: verify your account at https://paypa1.example/login"}'
```

The endpoint accepts a JSON object with a `text` string and returns the same structured analysis used by the interface. CORS is enabled for local extension development. Uvicorn is included in `requirements.txt`.

## Optional Transformers Backend

Set `JAWBREAKER_BACKEND=transformers` and configure the model and adapter IDs for a deployment that uses Transformers. The adapter is optional; the local GGUF path uses the base model only.

## Optional Local Hybrid Backend

The hybrid backend uses the specified four-class DistilBERT phishing-email/URL classifier as a CPU security signal and MiniCPM5-1B GGUF as the explanation model. The supplied classifier is DistilBERT, not DeBERTa. Neither model sends message contents to a hosted inference API. On the first run, Transformers downloads the classifier weights from Hugging Face; subsequent classification is local.

In PowerShell from the project root:

```powershell
$env:JAWBREAKER_BACKEND = "hybrid"
$env:JAWBREAKER_CLASSIFIER_MODEL_ID = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
$env:JAWBREAKER_PHISHING_THRESHOLD = "0.75"
$env:JAWBREAKER_SUSPICIOUS_THRESHOLD = "0.50"
..\venv\Scripts\python.exe app.py
```

The classifier score combines the checkpoint's two phishing classes. Scores at or above the phishing threshold raise risk to `dangerous`; scores at or above the suspicious threshold raise at least to `suspicious`. PhishBrake never maps these outputs to a new risk label, and the classifier cannot downgrade an existing dangerous verdict. The thresholds are configurable and are not an accuracy guarantee.

Project initiative: ML Empowerment Build Challenge 3.0. Maintainer: The Jawbreaker Team.
