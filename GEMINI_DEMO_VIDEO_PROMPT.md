# Gemini Prompt for PhishBrake Demo Video Generation

Use the prompt below with **Gemini 1.5 Pro / Gemini Advanced** (or any LLM/video generator like Veo/Sora/Runway) to generate a full video script, voiceover audio cues, scene-by-scene visual descriptions, and AI video generator prompts for your Devpost demo video.

---

### 📋 Copy & Paste Prompt for Gemini

```markdown
Act as an expert AI Video Producer, Creative Director, and Tech Product Presenter. 

I need a complete 2-minute video production package for my Devpost project submission: **PhishBrake – Private Scam Defense for Someone You Love** (submitted to the ML Empowerment Build Challenge 3.0).

---

### 📌 Project Background & Key Details
- **Product Name:** PhishBrake
- **Tagline:** Private scam defense for someone you love.
- **Origin Story:** Built for non-technical family members and grandparents who receive urgent, high-pressure SMS, email, and QR code scams demanding money or passwords.
- **Core Technology:** Local small-model AI inference (MiniCPM5-1B GGUF via llama-cpp-python + DistilBERT hybrid classifier). 100% private, zero cloud API leakage.
- **Key Features:** 
  1. Plain-English Verdict (DANGER, SUSPECT, CHECK, CLEAR).
  2. Scam DNA Breakdown (Imposter Identity, Pressure Tactic, Target Goal, Risk).
  3. Zero-Width Unicode Evasion & Domain Spoofing Detection.
  4. 1-Click Trusted Contact Note (copyable message for family help).
  5. Multi-surface ecosystem: Gradio Web App + Chrome Extension (Manifest V3) + QR Quishing Scanner.
- **ML Performance:** 100% safety recall (`dangerous_as_safe = 0` across 652 hard evaluation cases).

---

### 🎬 Video Specifications Required
1. **Target Duration:** 120 seconds (2 minutes).
2. **Tone:** Empathetic, crisp, modern, tech-forward, and authoritative.
3. **Structure & Scenes:**
   - **Scene 1 (0:00 - 0:20) - The Problem:** High-pressure scam message arriving on a smartphone (e.g., fake USPS fee alert, fake bank alert, fake family emergency text). The emotional anxiety it causes for a non-technical family member.
   - **Scene 2 (0:20 - 0:45) - Introducing PhishBrake:** Transition to the clean PhishBrake UI (Gradio web app & Chrome extension). Show direct text paste & 1-click scanning.
   - **Scene 3 (0:45 - 1:15) - The AI Engine & Safety Defense:** Highlight local MiniCPM5-1B GGUF inference, homograph/zero-width character stripping, zero-cloud privacy, and `dangerous_as_safe = 0` calibration.
   - **Scene 4 (1:15 - 1:40) - Extension & Multi-Surface Ecosystem:** Show the Chrome Extension scanning an email on a desktop browser and generating a copyable trusted contact note.
   - **Scene 5 (1:40 - 2:00) - Call to Action & Conclusion:** Recap privacy + safety + real-world impact for ML Empowerment Build Challenge 3.0. GitHub repo link callout.

---

### 🎯 Please Output the Following:
1. **Scene-by-Scene Script:** Voiceover script with exact timing (in seconds) and audio tone notes.
2. **Visual Cues & Screen Recording Directives:** What screen capture or UI animation to display on screen for each segment.
3. **AI Video Generator Prompts (Veo / Runway / Sora / Midjourney):** 5 specific text-to-video / text-to-image prompts to generate cinematic background B-roll for Scene 1 (the scam message problem) and Scene 5 (family safety ending).
4. **ElevenLabs / Text-to-Speech Prompt:** Recommended voice actor settings, tone, pacing, and emphasis rules.
```

---

### 💡 How to Use This Prompt
1. Copy the text box above into **Gemini Advanced** or **Gemini 1.5 Pro**.
2. Gemini will output a scene-by-scene production script, exact voiceover lines, visual cues, and image/video generator prompts.
3. Use the AI prompts in tools like **Runway Gen-3**, **Google Veo**, **Sora**, or **ElevenLabs** to assemble your final video.
