# PhishBrake Demo Video & Submission Walkthrough Guide

**Challenge:** ML Empowerment Build Challenge 3.0  
**Project:** PhishBrake – Private Scam Defense for Someone You Love  
**Live GitHub Repo:** [https://github.com/shankarsai000/PhishBrake](https://github.com/shankarsai000/PhishBrake)

---

## 🎬 2-Minute Video Narration Script (Devpost / YouTube)

### **[0:00 - 0:25] The Problem**
*(Show screen of a phone/inbox receiving an urgent SMS: "USPS: Your package is held due to an unpaid fee. Verify now: http://usps-track-secure.example")*

> **Voiceover:**  
> *"Every day, millions of people—especially grandparents and non-technical loved ones—receive high-pressure scam messages. From fake package delivery holds to urgent 'grandma in trouble' texts, scammers exploit urgency and fear to steal money and credentials. Existing cybersecurity tools are built for IT experts, not everyday families who just need a calm safety check."*

---

### **[0:25 - 1:00] The Solution: PhishBrake Web App**
*(Show PhishBrake Web UI on screen at `http://127.0.0.1:7860`)*

> **Voiceover:**  
> *"Meet PhishBrake—a private, local-first AI scam defense application built to protect the people you love. PhishBrake lets anyone paste a suspicious message or upload a screenshot or QR code."*

*(Click 'Check message' on the USPS package scam sample)*

> *"In seconds, PhishBrake breaks down the risk:  
> 1. A clear visual verdict: **CRITICAL: Scam Detected**.  
> 2. **Scam DNA:** Unpacking who they pretend to be, how they pressure you, what they want, and what could happen.  
> 3. **Safest Next Step:** Plain-English guidance telling the user exactly what to do—never sending them to suspicious links or phone numbers."*

---

### **[1:00 - 1:30] 1-Click Trusted Contact Note & Chrome Extension**
*(Demonstrate clicking the "COPY NOTE" button)*

> **Voiceover:**  
> *"PhishBrake includes a 1-click warning note generator. With one tap, users can copy a plain-English note tailored for a loved one to sanity-check the message together."*

*(Show Chrome Extension right-click context menu and popup)*

> *"We also built a Chrome Extension (Manifest V3). Highlight any text on any webpage or email, right-click, and choose 'Scan with PhishBrake' for instant local analysis."*

---

### **[1:30 - 2:00] Privacy, ML Engineering & Impact**
*(Show terminal running `pytest` and `eval/run_eval.py` passing 100%)*

> **Voiceover:**  
> *"PhishBrake runs 100% locally using MiniCPM5-1B GGUF and a DistilBERT hybrid classifier—keeping private conversations completely on-device without cloud API dependencies. Our deterministic safety guard achieves zero dangerous false-negatives across hard evaluation benchmarks.*  
> *PhishBrake turns high-pressure scam moments into calm, safe verification. Thank you!"*

---

## 📷 Screenshots Included in Repository

1. **USPS Package Scam Breakdown:**  
   `docs/phishbrake_usps_demo.png`  
   *(View on GitHub: [phishbrake_usps_demo.png](https://github.com/shankarsai000/PhishBrake/raw/main/docs/phishbrake_usps_demo.png))*

2. **Family Emergency Scam Breakdown:**  
   `docs/phishbrake_family_demo.png`  
   *(View on GitHub: [phishbrake_family_demo.png](https://github.com/shankarsai000/PhishBrake/raw/main/docs/phishbrake_family_demo.png))*

---

## 🚀 How to Run the App for Demo Recording

```powershell
# 1. Start the PhishBrake App Server
cd c:\jawbreaker-main\jawbreaker-main
$env:JAWBREAKER_BACKEND="heuristic"
..\venv\Scripts\python.exe app.py

# 2. Open browser at http://127.0.0.1:7860
```
