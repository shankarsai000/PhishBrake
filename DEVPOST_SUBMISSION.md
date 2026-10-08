# Devpost Submission Draft: PhishBrake

**Challenge:** ML Empowerment Build Challenge 3.0  
**Project Title:** PhishBrake – Private Scam Defense for Someone You Love  
**Category:** Machine Learning / AI | Social Good | Beginner Friendly  
**Repository:** [GitHub / Project Files]

---

## 1. Project Description

### 💡 Problem Statement
Digital scams, SMS phishing (smishing), QR code scams (quishing), and impersonation fraud are skyrocketing. Vulnerable individuals—especially grandparents, non-technical family members, and students—are targeted daily with urgent, high-pressure messages demanding money, passwords, or verification codes. Existing cybersecurity tools are built for IT experts, not everyday people who simply need a calm, plain-English check on whether to reply, click, or ask for help.

### 🛡️ Solution Overview: PhishBrake
PhishBrake is a private, local-first AI scam defense application designed to shield loved ones from digital fraud. Users simply paste any suspicious text message, email, DM, or QR code. PhishBrake runs local AI inference directly on the device—protecting user privacy without sending private chats to cloud APIs—and delivers:
1. **Clear Risk Verdict:** Immediate visual indication (DANGER, SUSPECT, CHECK, CLEAR).
2. **Plain-English Explanation:** What the sender is pretending to be and what manipulation tactics are being used.
3. **Scam DNA Breakdown:** Unpacks *Who they pretend to be*, *How they pressure you*, *What they want*, and *What could happen*.
4. **Safest Next Step:** One clear, actionable directive (never telling the user to click or call unverified numbers).
5. **1-Click Trusted Contact Note:** Generates a copyable note tailored for a loved one ("Hi Grandma / Mom, can you check this message with me...").

---

## 2. Key Features

- **Local Small-Model AI Inference:** Powered by MiniCPM5-1B GGUF via `llama-cpp-python` and a DistilBERT hybrid classifier—100% private, running locally on CPU/GPU.
- **Deterministic Safety Guard:** Schema validation and deterministic heuristic rules ensure **0 dangerous false-negatives** (`dangerous_as_safe = 0`).
- **Stealth Evasion & Homograph Audit:** Automatically strips and flags zero-width unicode character evasion (`\u200b`), UserInfo `@` domain spoofing (`https://paypal.com@evil.com`), raw IP hosts, and URL shorteners.
- **Chrome Extension (Manifest V3):** Right-click any selected text on any webpage or email to "Scan with PhishBrake", or use the interactive extension popup.
- **QR Code (Quishing) Scanner:** Upload suspicious screenshots or QR codes to extract and audit hidden link payloads.
- **Session Memory & Instant Edge Cache:** Remembers recent scam patterns in session to help users spot recurring attack tactics.

---

## 3. Technologies Used

- **AI / ML & NLP:** PyTorch, Hugging Face Transformers, `llama-cpp-python`, MiniCPM5-1B GGUF, DistilBERT (`phishing-email-detection-distilbert_v2.4.1`).
- **Backend & Web Server:** Python 3.14 / 3.12, Gradio 6.16, FastAPI, Uvicorn, PyTest (74 automated tests).
- **Browser Extension:** React, Vite, JavaScript, HTML5, CSS3, Chrome Extension Manifest V3 API.
- **Data & Evaluation:** Custom synthetic dataset calibration pipeline (`generate_v9_data.py`), 652-case hard evaluation benchmark (`hard_v9_eval.jsonl`), Modal A100 training scripts.

---

## 4. Target Users

- **Families & Seniors:** Non-technical individuals and elderly relatives who need a calm, accessible safety check before replying to suspicious messages.
- **Students & Everyday Users:** Anyone receiving unknown texts, job offer scams, package delivery hold alerts, or suspicious links.

---

## 5. Why PhishBrake Can Win ML Empowerment Build Challenge 3.0

1. **High Social Impact:** Solves a major real-world problem (protecting families from devastating scam fraud) with a human-centered, compassionate design.
2. **Privacy-First AI:** Demonstrates how compact 1B models can deliver state-of-the-art safety locally on consumer devices without cloud API dependencies.
3. **Complete Multi-Surface Ecosystem:** Web App + Chrome Extension + QR Scanner + Automated Test & Evaluation Harness.
4. **Rigorous ML Engineering & Honest Claims:** Includes transparent evaluation evidence (`HONEST_SUBMISSION.md`, `DEVELOPMENT_EVIDENCE.md`), 74 unit tests, and multi-version dataset calibration.
