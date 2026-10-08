import os
import json
from base64 import b64decode
from dataclasses import asdict
from functools import lru_cache
from html import escape
from io import BytesIO
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

from jawbreaker.analyzers import (
    DEFAULT_PHISHING_CLASSIFIER_ID,
    analysis_to_prediction,
    build_hybrid_analyzer,
    build_llama_cpp_analyzer,
    build_transformers_analyzer,
    heuristic_analyzer,
    prediction_to_analysis,
    repair_prediction,
    should_apply_heuristic_guard,
    validate_prediction,
)
from jawbreaker.cache import lookup_cache, save_cache
from jawbreaker.homograph import audit_message_urls
from jawbreaker.quishing import extract_qr_payload
from jawbreaker.render import render_analysis_html, render_memory_html, render_scanning_html
from jawbreaker.schema import ScamAnalysis
from jawbreaker.trust import (
    build_trusted_note,
    confidence_metadata,
    is_low_context_message,
    low_context_analysis,
)

try:
    import spaces
except ImportError:
    spaces = None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    text: str


@app.post("/api/scan")
def scan(request: ScanRequest) -> dict[str, Any]:
    return analysis_payload(request.text)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "backend": current_backend(), "model": model_status_label()}


EXAMPLES = [
    "USPS: Your package is held due to an unpaid fee. Verify now: http://usps-track-secure.example",
    "Hi Grandma, I lost my phone. This is my new number. Can you send $800 for rent today? Please don't tell Mom.",
    "Coinbase alert: We received a request to update the phone number on your account. If this wasn't you, call support immediately at [callback number].",
    "Hi! I'm a recruiter from TikTok Shop. We are looking for a part-time assistant. Flexible remote work, 60 minutes per day, $330-$750 per day. Contact me on WhatsApp at [phone number].",
    "Your dentist appointment is confirmed for Tuesday at 2:00 PM. Reply C to confirm or R to reschedule.",
    "Your verification code is 482913. Do not share this code with anyone. We will never ask for it by phone.",
    "PayPal account notice: Verify at https://paypa1.example/login to restore access.",
    "Cache demo: Urgent bank alert. Send $250 and share your 6-digit verification code to prevent account lockout. Call 415-555-0199.",
]

DEFAULT_TRANSFORMERS_MODEL_ID = "openbmb/MiniCPM5-1B"
DEFAULT_ADAPTER_ID = "build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8"
DEFAULT_TRANSFORMERS_MAX_TOKENS = 512
CACHE_HIT_STATUS = "⚡ Instant Edge Cache Hit"
LOGO_PATH = Path("phishbrake_logo.png")
SAFE_QR_DEMO_PATH = Path("qr_demo_safe.png")
PHISHING_QR_DEMO_PATH = Path("qr_demo_phishing.png")

FORCE_LIGHT_JS = """() => {
    const forceLight = () => document.body.classList.remove('dark');
    forceLight();
    new MutationObserver(forceLight).observe(document.body, {
        attributes: true,
        attributeFilter: ['class'],
    });
}"""

FORCE_LIGHT_HEAD = """
<script>
(() => {
  const forceLight = () => {
    if (document.body) document.body.classList.remove("dark");
  };
  const copyPlan = async (button) => {
    const source = button.dataset.copy || "";
    if (!source) return;
    await navigator.clipboard.writeText(source);
    const original = button.textContent || "COPY NOTE";
    button.textContent = "COPIED";
    window.setTimeout(() => {
      button.textContent = original;
    }, 1200);
  };
  document.addEventListener("click", (event) => {
    const button = event.target.closest?.(".inline-copy-btn");
    if (!button) return;
    event.preventDefault();
    copyPlan(button);
  });
  window.addEventListener("DOMContentLoaded", () => {
    forceLight();
    new MutationObserver(forceLight).observe(document.body, {
      attributes: true,
      attributeFilter: ["class"],
    });
  });
})();
</script>
"""


def app_theme() -> gr.Theme:
    theme = gr.themes.Soft(
        primary_hue="red",
        secondary_hue="slate",
        neutral_hue="zinc",
        radius_size="sm",
    )
    theme.set(
        block_label_background_fill="transparent",
        block_label_border_color="transparent",
        block_label_text_color="#6b5144",
        block_label_text_size="13px",
        block_label_text_weight="700",
    )
    return theme


def app_css() -> str:
    return Path("style.css").read_text(encoding="utf-8")


def gpu_callback(fn):
    if spaces is None:
        return fn
    return spaces.GPU(fn)


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool | None = None) -> bool | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_adapter_id(model_id: str) -> str | None:
    configured = os.getenv("JAWBREAKER_ADAPTER_ID")
    if configured is not None:
        return configured or None
    if model_id == DEFAULT_TRANSFORMERS_MODEL_ID:
        return DEFAULT_ADAPTER_ID
    return None


def _llama_cpp_options() -> dict[str, Any]:
    return {
        "chat_format": os.getenv("JAWBREAKER_CHAT_FORMAT") or None,
        "n_ctx": _env_int("JAWBREAKER_N_CTX", 1024) or 1024,
        "n_threads": _env_int("JAWBREAKER_N_THREADS", 2),
        "n_gpu_layers": _env_int("JAWBREAKER_N_GPU_LAYERS", 0) or 0,
        "n_batch": _env_int("JAWBREAKER_N_BATCH", 128) or 128,
        "n_ubatch": _env_int("JAWBREAKER_N_UBATCH", 128) or 128,
        "offload_kqv": _env_bool("JAWBREAKER_OFFLOAD_KQV", True),
        "op_offload": _env_bool("JAWBREAKER_OP_OFFLOAD"),
        "max_tokens": _env_int("JAWBREAKER_MAX_TOKENS", 256) or 256,
        "temperature": float(os.getenv("JAWBREAKER_TEMPERATURE", "0")),
    }


@lru_cache(maxsize=1)
def get_analyzer():
    backend = current_backend()
    if backend == "heuristic":
        return heuristic_analyzer

    if backend == "llama-cpp":
      return build_llama_cpp_analyzer(resolve_model_path(), **_llama_cpp_options())

    if backend == "hybrid":
      return build_hybrid_analyzer(
        resolve_model_path(),
        classifier_model_id=os.getenv(
          "JAWBREAKER_CLASSIFIER_MODEL_ID",
          DEFAULT_PHISHING_CLASSIFIER_ID,
        ),
        phishing_threshold=float(os.getenv("JAWBREAKER_PHISHING_THRESHOLD", "0.75")),
        suspicious_threshold=float(os.getenv("JAWBREAKER_SUSPICIOUS_THRESHOLD", "0.50")),
        **_llama_cpp_options(),
      )

    if backend in {"transformers", "zerogpu"}:
        model_id = os.getenv("JAWBREAKER_TRANSFORMERS_MODEL_ID", DEFAULT_TRANSFORMERS_MODEL_ID)
        return build_transformers_analyzer(
            model_id,
            adapter_id=default_adapter_id(model_id),
            max_new_tokens=_env_int("JAWBREAKER_MAX_TOKENS", DEFAULT_TRANSFORMERS_MAX_TOKENS)
            or DEFAULT_TRANSFORMERS_MAX_TOKENS,
            temperature=float(os.getenv("JAWBREAKER_TEMPERATURE", "0")),
            device_map=os.getenv("JAWBREAKER_DEVICE_MAP", "auto"),
            dtype=os.getenv("JAWBREAKER_TORCH_DTYPE", "auto"),
            trust_remote_code=_env_bool("JAWBREAKER_TRUST_REMOTE_CODE", True) or False,
            attn_implementation=os.getenv("JAWBREAKER_ATTENTION_IMPLEMENTATION", "eager") or None,
        )

    raise ValueError(f"Unsupported JAWBREAKER_BACKEND: {backend}")


def current_backend() -> str:
    return os.getenv("JAWBREAKER_BACKEND", "llama-cpp").strip().lower()


def should_warm_model() -> bool:
    return _env_bool("JAWBREAKER_WARM_MODEL", True) and current_backend() in {"transformers", "zerogpu"}


def resolve_model_path() -> Path:
    model_path = Path(os.getenv("JAWBREAKER_MODEL_PATH", "models/minicpm5-1b-gguf/minicpm5-1b-Q4_K_M.gguf"))
    if model_path.exists():
        return model_path

    repo_id = os.getenv("JAWBREAKER_MODEL_REPO", "Abiray/MiniCPM5-1B-GGUF")
    filename = os.getenv("JAWBREAKER_MODEL_FILE", "minicpm5-1b-Q4_K_M.gguf")
    cache_dir = os.getenv("JAWBREAKER_MODEL_CACHE")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download the configured model file.") from exc

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir))


def run_analysis(message: str, memory: list[dict] | None) -> ScamAnalysis:
    memory = memory or []
    heuristic = ScamAnalysis.from_heuristics(message, memory)
    homograph_detected = bool(audit_message_urls(message)["flagged_urls"])
    if homograph_detected:
        heuristic = prediction_to_analysis(
            repair_prediction(
                analysis_to_prediction(heuristic),
                homograph_detected=True,
            ),
            similar_memory=heuristic.similar_memory,
        )
    started = perf_counter()
    try:
        prediction = get_analyzer()(message)
    except Exception as exc:
        print(
            "jawbreaker analyzer_error fallback=heuristic "
            f"elapsed={perf_counter() - started:.2f}s error={exc!r}",
            flush=True,
        )
        heuristic.summary = f"{heuristic.summary} PhishBrake used its safety fallback because the model response was not usable."
        return heuristic
    raw_validation_errors = validate_prediction(prediction)
    prediction = repair_prediction(prediction)
    validation_errors = validate_prediction(prediction)
    model_analysis = prediction_to_analysis(prediction, similar_memory=heuristic.similar_memory)
    print(f"jawbreaker analyze elapsed={perf_counter() - started:.2f}s", flush=True)
    if should_use_heuristic_guard(model_analysis, heuristic, validation_errors, message):
        print(
            "jawbreaker guard=heuristic "
            f"model_risk={model_analysis.risk_level} heuristic_risk={heuristic.risk_level}",
            flush=True,
        )
        return heuristic
    if raw_validation_errors:
        print(f"jawbreaker schema_repaired errors={raw_validation_errors}", flush=True)
    return model_analysis


def warm_model_impl() -> None:
    backend = current_backend()
    if not should_warm_model():
        print(f"jawbreaker warm_model skip backend={backend}", flush=True)
        return None

    started = perf_counter()
    print(f"jawbreaker warm_model start backend={backend}", flush=True)
    get_analyzer()
    print(f"jawbreaker warm_model ready elapsed={perf_counter() - started:.2f}s", flush=True)
    return None


@gpu_callback
def warm_model() -> None:
    return warm_model_impl()


def memory_entry_from_analysis(message: str, analysis: ScamAnalysis) -> dict[str, Any]:
    return {
        "summary": analysis.summary,
        "scam_type": analysis.scam_type,
        "risk_level": analysis.risk_level,
        "fingerprint": analysis.scam_dna,
        "text": message[:240],
    }


def render_current_memory(memory: list[dict]) -> str:
    return render_memory_html(ScamAnalysis.from_heuristics("", memory), memory)


def render_save_status(message: str = "") -> str:
    if not message:
        return "<div class='save-status'></div>"
    return f"<div class='save-status'>{message}</div>"


def model_status_label() -> str:
    if current_backend() == "hybrid":
      return "Hybrid (DistilBERT + MiniCPM5-1B)"
    if current_backend() == "llama-cpp":
        default_path = "models/minicpm5-1b-gguf/minicpm5-1b-Q4_K_M.gguf"
        model_file = os.getenv("JAWBREAKER_MODEL_FILE") or Path(
            os.getenv("JAWBREAKER_MODEL_PATH", default_path)
        ).name
        model_name = Path(model_file).stem.replace("-", " ")
        if model_name.lower().startswith("minicpm5 1b"):
            return "MiniCPM5-1B GGUF (base)"
        return f"{model_name} GGUF"

    model_id = os.getenv("JAWBREAKER_TRANSFORMERS_MODEL_ID", DEFAULT_TRANSFORMERS_MODEL_ID)
    adapter_id = default_adapter_id(model_id)
    if model_id == "openbmb/MiniCPM5-1B" and adapter_id == "build-small-hackathon/jawbreaker-minicpm5-1b-lora-v8":
        return "MiniCPM5-1B + PhishBrake LoRA v8"
    if adapter_id:
        return adapter_id.rsplit("/", 1)[-1].replace("jawbreaker-", "").upper()
    return model_id.rsplit("/", 1)[-1].upper()


def backend_status_label() -> str:
    backend = current_backend()
    if backend == "hybrid":
        return "LOCAL HYBRID"
    if backend == "zerogpu":
        return "ZEROGPU"
    if backend == "transformers":
        return "TRANSFORMERS"
    if backend == "llama-cpp":
        return "LLAMA.CPP"
    if backend == "heuristic":
        return "LOCAL FALLBACK"
    return backend.upper()


def _cached_analysis(text: str) -> ScamAnalysis | None:
    cached = lookup_cache(text)
    if cached is None:
        return None
    try:
        return ScamAnalysis(**cached)
    except (TypeError, ValueError):
        return None


def _save_confident_analysis(text: str, analysis: ScamAnalysis) -> None:
    if text.strip() and analysis.risk_level != "needs_check":
        save_cache(text, asdict(analysis))


def analysis_payload(message: str, memory: list[dict] | None = None) -> dict[str, Any]:
    memory = memory or []
    started = perf_counter()
    analysis = _cached_analysis(message) if message.strip() else None
    cache_hit = analysis is not None
    if analysis is None and not message.strip():
        analysis = ScamAnalysis.from_heuristics(message, memory)
    elif analysis is None and is_low_context_message(message):
        analysis = low_context_analysis(message, memory)
    elif analysis is None:
        analysis = run_analysis(message, memory)
    if not cache_hit:
        _save_confident_analysis(message, analysis)

    entry = memory_entry_from_analysis(message, analysis)
    if message.strip() and (not memory or memory[-1].get("text") != entry.get("text")):
        memory.append(entry)

    payload = {
        "analysis": asdict(analysis),
        "confidence": confidence_metadata(message, analysis),
        "copy_plan": build_handoff_message(message, analysis),
        "memory": memory[-6:],
        "message": message,
        "model_label": model_status_label(),
        "elapsed_seconds": round(perf_counter() - started, 2),
    }
    if cache_hit:
        payload["cache_status"] = CACHE_HIT_STATUS
    return payload


def _image_from_data_url(image_data: str | None) -> Image.Image | None:
    if not image_data:
        return None
    header, separator, encoded_image = image_data.partition(",")
    if not separator or not header.startswith("data:image/"):
        return None
    try:
        with Image.open(BytesIO(b64decode(encoded_image, validate=True))) as image:
            return image.copy()
    except (OSError, ValueError):
        return None


def _message_with_qr_payload(message: str, image: Any | None) -> tuple[str, str | None]:
    qr_text = extract_qr_payload(image) if image is not None else None
    if not qr_text:
        return message, None
    return f"{message}\n[Detected QR Code Link]: {qr_text}", qr_text


def logo_data_uri() -> str:
    import base64

    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def qr_demo_data_uri(image_path: Path) -> str:
    import base64

    if not image_path.exists():
        return ""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def paper_shield_html() -> str:
    examples_json = json.dumps(EXAMPLES).replace("</", "<\\/")
    logo_uri = logo_data_uri()
    model_label = escape(model_status_label())
    backend_label = escape(backend_status_label())
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PhishBrake</title>
  <style>
    /* Hallmark · macrostructure: Workbench · genre: playful · theme: custom family-safety report
     * pre-emit critique: P5 H4 E4 S5 R4 V4 · contrast: pass target (40-41)
     */
    :root {{
      color-scheme: light;
      --color-bg: #f7f1e4;
      --color-bg-warm: #fbf5e8;
      --color-paper: #fffdf7;
      --color-paper-soft: #fff8ea;
      --color-ink: #17130f;
      --color-muted: #66594e;
      --color-rule: #241a13;
      --color-focus: #246bfe;
      --color-action: #f4c400;
      --color-action-ink: #17130f;
      --color-danger: #d94d4b;
      --color-danger-soft: #ffe1df;
      --color-danger-ink: #17130f;
      --color-coral: #f58b72;
      --color-fold: #f0e4cc;
      --color-terminal-good: #80f2a8;
      --color-terminal-muted: #9f9a92;
      --color-safe: #238756;
      --color-safe-soft: #daf5e4;
      --color-check: #af6b11;
      --color-check-soft: #fff0ca;
      --shadow-soft: 0 18px 45px rgba(36, 26, 19, .12);
      --shadow-hard: 5px 5px 0 var(--color-rule);
      --radius: 8px;
      --font-label: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --font-display: Georgia, "Times New Roman", Times, serif;
      --font-body: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    * {{ box-sizing: border-box; }}
    html {{
      overflow-x: clip;
      background: var(--color-bg);
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      overflow-x: clip;
      background:
        linear-gradient(135deg, rgba(36, 26, 19, .035) 0 1px, transparent 1px 28px),
        radial-gradient(circle at 8% 4%, rgba(244, 196, 0, .18), transparent 24rem),
        var(--color-bg);
      color: var(--color-ink);
      font-family: var(--font-body);
    }}

    button, textarea {{ font: inherit; }}
    button {{ cursor: pointer; }}

    .shell {{
      width: min(1320px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 56px;
    }}

    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding-bottom: 18px;
      border-bottom: 2px solid var(--color-rule);
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }}

    .brand img {{
      width: 44px;
      height: 44px;
      border: 2px solid var(--color-rule);
      border-radius: 8px;
      background: var(--color-paper);
    }}

    .brand h1 {{
      margin: 0;
      font-family: var(--font-display);
      font-size: clamp(1.55rem, 3vw, 2.35rem);
      line-height: .92;
      letter-spacing: 0;
      min-width: 0;
      overflow-wrap: anywhere;
    }}

    .brand-copy {{
      display: grid;
      gap: 4px;
      min-width: 0;
    }}

    .tagline {{
      margin: 0;
      color: var(--color-muted);
      font-size: clamp(.86rem, 1vw, .96rem);
      line-height: 1.25;
    }}

    .status {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      font-family: var(--font-label);
      font-weight: 800;
      font-size: .72rem;
      text-align: right;
    }}

    .status > span {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border: 1.5px solid var(--color-rule);
      border-radius: 999px;
      background: var(--color-paper-soft);
      padding: 6px 10px;
      white-space: nowrap;
    }}

    .hero {{
      display: grid;
      gap: 7px;
      padding: 18px 0 16px;
    }}

    .hero h2 {{
      margin: 0;
      font-family: var(--font-display);
      font-size: clamp(1.45rem, 2.1vw, 1.9rem);
      line-height: 1;
      letter-spacing: 0;
      max-width: 100%;
      min-width: 0;
      white-space: nowrap;
    }}

    .hero p {{
      margin: 0;
      color: var(--color-muted);
      font-size: clamp(.8rem, .9vw, .9rem);
      line-height: 1.35;
      max-width: none;
      white-space: nowrap;
    }}

    .safety-note {{
      width: fit-content;
      max-width: 100%;
      margin-top: 4px;
      border: 1.5px solid var(--color-rule);
      border-radius: 999px;
      background: var(--color-paper-soft);
      padding: 6px 10px;
      color: var(--color-muted);
      font-family: var(--font-label);
      font-size: .62rem;
      font-weight: 800;
      line-height: 1.25;
      white-space: normal;
    }}

    .privacy-note {{
      margin: -3px 0 12px;
      color: var(--color-muted);
      font-family: var(--font-label);
      font-size: .66rem;
      font-weight: 800;
      line-height: 1.35;
    }}

    .workspace {{
      display: grid;
      grid-template-columns: minmax(320px, 430px) minmax(0, 1fr);
      gap: 36px;
      align-items: start;
    }}

    .paper {{
      position: relative;
      background: var(--color-paper);
      border: 2px solid var(--color-rule);
      border-radius: var(--radius);
      box-shadow: var(--shadow-soft);
    }}

    .paper::after {{
      display: none;
    }}

    .note {{
      padding: 28px;
    }}

    .section-label {{
      margin: 0 0 13px;
      font-family: var(--font-label);
      font-size: .86rem;
      font-weight: 900;
      text-transform: uppercase;
    }}

    textarea {{
      width: 100%;
      min-height: 255px;
      resize: vertical;
      border: 2px solid var(--color-rule);
      border-radius: 4px;
      background:
        linear-gradient(var(--color-paper) 31px, rgba(36, 26, 19, .08) 32px),
        var(--color-paper);
      background-size: 100% 32px;
      color: var(--color-ink);
      padding: 16px;
      font-size: .92rem;
      line-height: 1.55;
      outline: 2px solid transparent;
      outline-offset: 2px;
    }}

    textarea:focus-visible {{
      outline-color: var(--color-focus);
      box-shadow: 0 0 0 4px rgba(36, 107, 254, .12);
    }}

    .run {{
      width: 100%;
      margin-top: 18px;
      min-height: 58px;
      border: 2px solid var(--color-rule);
      border-radius: 999px;
      background: var(--color-action);
      color: var(--color-action-ink);
      font-family: var(--font-label);
      font-weight: 950;
      font-size: .94rem;
      text-transform: uppercase;
      white-space: nowrap;
      box-shadow: 0 5px 0 var(--color-rule);
      transition: transform .16s ease, box-shadow .16s ease, opacity .16s ease;
    }}

    .run:focus-visible,
    .copy-btn:focus-visible,
    .sample:focus-visible {{
      outline: 2px solid var(--color-focus);
      outline-offset: 3px;
    }}

    .run:active {{
      transform: translateY(3px);
      box-shadow: 0 2px 0 var(--color-rule);
    }}

    .run[disabled] {{
      opacity: .55;
      cursor: wait;
    }}

    .samples {{
      margin-top: 22px;
      display: grid;
      gap: 9px;
    }}

    .sample {{
      border: 1.5px solid var(--color-rule);
      border-radius: 999px;
      background: var(--color-paper-soft);
      padding: 10px 14px;
      text-align: left;
      font-size: .95rem;
      line-height: 1.22;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .result-stack {{
      display: grid;
      gap: 24px;
    }}

    .result-top {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
      align-items: stretch;
    }}

    .result-top .card {{
      min-height: 100%;
    }}

    .card {{
      position: relative;
      border: 2px solid var(--color-rule);
      border-radius: var(--radius);
      background: var(--color-paper);
      box-shadow: var(--shadow-soft);
      overflow: hidden;
    }}

    .card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 42px;
      padding: 10px 18px;
      border-bottom: 2px solid var(--color-rule);
      background: var(--color-paper-soft);
      color: var(--color-ink);
      font-family: var(--font-label);
      font-weight: 900;
      font-size: .78rem;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }}

    .card-body {{
      padding: clamp(16px, 2.4vw, 24px);
    }}

    .result-top .card-body {{
      padding: clamp(16px, 2.4vw, 24px);
    }}

    .standby-card .card-body {{
      min-height: 260px;
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(240px, .95fr);
      gap: clamp(24px, 4vw, 46px);
      align-items: center;
      text-align: left;
      background:
        linear-gradient(90deg, transparent 49%, rgba(36,26,19,.07) 50%, transparent 51%),
        linear-gradient(0deg, transparent 49%, rgba(36,26,19,.07) 50%, transparent 51%),
        var(--color-paper);
      background-size: 42px 42px;
    }}

    .shield-mark {{
      width: min(112px, 30vw);
      aspect-ratio: 1;
      margin: 0 0 16px;
      border: 2px solid var(--color-rule);
      border-radius: 50%;
      background:
        radial-gradient(circle, var(--color-danger) 0 22%, var(--color-paper-soft) 23% 38%, var(--color-coral) 39% 56%, var(--color-paper-soft) 57% 72%, var(--color-rule) 73% 100%);
      box-shadow: var(--shadow-hard);
    }}

    .standby-card h3 {{
      margin: 0 0 12px;
      max-width: 22ch;
      font-family: var(--font-display);
      font-size: clamp(1.3rem, 2vw, 1.75rem);
      line-height: 1.04;
      min-width: 0;
      overflow-wrap: anywhere;
    }}

    .standby-card p {{
      margin: 0;
      max-width: 34rem;
      color: var(--color-muted);
      line-height: 1.5;
      font-size: .84rem;
    }}

    .standby-list {{
      display: grid;
      gap: 10px;
      margin-top: 16px;
      padding: 0;
      list-style: none;
    }}

    .standby-list li {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      max-width: 34rem;
    }}

    .standby-list strong {{
      display: inline-grid;
      place-items: center;
      width: 1.45rem;
      height: 1.45rem;
      border: 1.5px solid var(--color-rule);
      border-radius: 999px;
      background: var(--color-action);
      font-family: var(--font-label);
      font-size: .84rem;
    }}

    .standby-preview {{
      display: grid;
      gap: 12px;
      align-self: stretch;
    }}

    .standby-tile {{
      display: grid;
      gap: 6px;
      align-content: center;
      min-height: 72px;
      border: 2px solid var(--color-rule);
      border-radius: 6px;
      background: var(--color-paper-soft);
      padding: 14px;
    }}

    .standby-tile strong {{
      font-family: var(--font-label);
      font-size: .76rem;
      text-transform: uppercase;
    }}

    .standby-tile span {{
      color: var(--color-muted);
      font-size: .9rem;
      line-height: 1.3;
    }}

    .loading-card .card-body {{
      min-height: 300px;
      background: var(--color-rule);
      color: var(--color-paper);
      font-family: var(--font-label);
    }}

    .loading-card .card-head {{
      background: var(--color-rule);
      color: var(--color-paper);
      border-bottom-color: var(--color-rule);
    }}

    .fold-loader {{
      display: grid;
      gap: 18px;
    }}

    .fold-bar {{
      height: 18px;
      border: 2px solid var(--color-paper);
      background: linear-gradient(90deg, var(--color-action) var(--progress, 12%), transparent 0);
      box-shadow: 4px 4px 0 rgba(0,0,0,.35);
    }}

    .loading-line {{
      margin: 0;
      font-weight: 900;
      color: var(--color-terminal-good);
    }}

    .loading-line.muted {{ color: var(--color-terminal-muted); }}

    .verdict-card.dangerous .stamp {{ background: var(--color-danger); color: var(--color-paper); }}
    .verdict-card.suspicious .stamp,
    .verdict-card.needs_check .stamp {{ background: var(--color-check); color: var(--color-paper); }}
    .verdict-card.safe .stamp {{ background: var(--color-safe); color: var(--color-paper); }}

    .verdict-layout {{
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 20px;
      align-items: center;
    }}

    .stamp {{
      width: 68px;
      height: 68px;
      display: grid;
      place-items: center;
      border: 3px solid var(--color-rule);
      border-radius: 50%;
      font-family: var(--font-label);
      font-size: .54rem;
      font-weight: 950;
      text-align: center;
      text-transform: uppercase;
      transform: rotate(-9deg);
      box-shadow: var(--shadow-hard);
    }}

    .verdict-title {{
      margin: 0 0 10px;
      font-family: var(--font-display);
      font-size: clamp(1.2rem, 1.75vw, 1.55rem);
      line-height: 1.03;
      overflow-wrap: anywhere;
    }}

    .summary {{
      margin: 0;
      font-size: .84rem;
      line-height: 1.45;
    }}

    .action-card {{
      background: var(--color-safe-soft);
      width: min(100%, 920px);
      justify-self: start;
    }}

    .action-card .card-body {{
      display: grid;
      gap: 16px;
      padding: clamp(20px, 3vw, 30px);
    }}

    .action-title {{
      margin: 0;
      font-family: var(--font-label);
      font-weight: 950;
      text-transform: uppercase;
      font-size: .8rem;
    }}

    .action-text {{
      margin: 0;
      font-family: var(--font-display);
      font-weight: 800;
      font-size: clamp(.94rem, 1.1vw, 1.05rem);
      line-height: 1.18;
      overflow-wrap: anywhere;
    }}

    .dna-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}

    .dna-item {{
      min-height: 70px;
      padding: 10px;
      border: 2px solid var(--color-rule);
      border-radius: 4px;
      background: linear-gradient(135deg, var(--color-paper) 0 78%, var(--color-fold) 79%);
    }}

    .dna-label {{
      margin-bottom: 9px;
      font-family: var(--font-label);
      font-size: .64rem;
      font-weight: 950;
      text-transform: uppercase;
    }}

    .dna-value {{
      font-family: var(--font-display);
      font-size: clamp(.78rem, .9vw, .88rem);
      font-weight: 800;
      line-height: 1.12;
      overflow-wrap: anywhere;
    }}

    .tactics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 16px;
    }}

    .tag {{
      border: 1.5px solid var(--color-rule);
      background: var(--color-danger);
      color: var(--color-paper);
      padding: 6px 9px;
      border-radius: 999px;
      font-family: var(--font-label);
      font-size: .72rem;
      font-weight: 950;
      text-transform: uppercase;
    }}

    .copy-card {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 160px;
      gap: 14px;
      align-items: stretch;
    }}

    .copy-note {{
      max-height: 136px;
      overflow: auto;
      white-space: pre-wrap;
      border: 2px solid var(--color-rule);
      border-radius: 4px;
      background: var(--color-paper);
      padding: 14px;
      font-size: .78rem;
      line-height: 1.38;
    }}

    .copy-btn {{
      border: 2px solid var(--color-rule);
      border-radius: 4px;
      background: var(--color-paper);
      font-family: var(--font-label);
      font-weight: 950;
      font-size: .78rem;
      text-transform: uppercase;
      box-shadow: var(--shadow-hard);
      white-space: nowrap;
    }}

    .memory {{
      margin-top: 28px;
      padding: 18px;
    }}

    .memory h3 {{
      margin: 0 0 12px;
      font-family: var(--font-label);
      font-size: .86rem;
      text-transform: uppercase;
    }}

    .memory-list {{
      display: grid;
      gap: 10px;
    }}

    .memory-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      padding: 12px;
      border: 1.5px solid var(--color-rule);
      border-radius: 4px;
      background: var(--color-danger-soft);
      font-size: .76rem;
      line-height: 1.28;
    }}

    .memory-row.safe {{
      background: var(--color-safe-soft);
    }}

    .memory-row.needs_check,
    .memory-row.suspicious {{
      background: var(--color-check-soft);
    }}

    .memory-badge {{
      padding: 5px 7px;
      border: 1.5px solid var(--color-rule);
      background: var(--color-danger);
      color: var(--color-paper);
      font-family: var(--font-label);
      font-size: .68rem;
      font-weight: 950;
      text-transform: uppercase;
    }}

    .memory-badge.safe {{
      background: var(--color-safe);
    }}

    .memory-badge.needs_check,
    .memory-badge.suspicious {{
      background: var(--color-check);
    }}

    .footer-strip {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
      gap: 8px 12px;
      margin-top: 30px;
      padding: 14px 18px;
      border-top: 2px solid var(--color-rule);
      color: var(--color-muted);
      font-family: var(--font-label);
      font-size: .78rem;
      font-weight: 800;
      text-align: center;
    }}

    .footer-strip strong {{
      color: var(--color-ink);
    }}

    .footer-dot {{
      color: var(--color-danger);
    }}

    .error {{
      border: 2px solid var(--color-rule);
      background: var(--color-danger-soft);
      padding: 14px;
      border-radius: 4px;
      font-weight: 800;
    }}

    @media (max-width: 1240px) {{
      .dna-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 860px) {{
      .shell {{
        width: min(100% - 22px, 620px);
        padding-top: 18px;
      }}

      .topbar {{
        align-items: flex-start;
        flex-direction: column;
        gap: 14px;
      }}

      .tagline {{
        font-size: .9rem;
      }}

      .status {{
        text-align: left;
        justify-content: flex-start;
      }}

      .status > span {{
        white-space: normal;
      }}

      .hero {{
        gap: 16px;
        padding: 26px 0 22px;
      }}

      .hero h2 {{
        max-width: 100%;
        white-space: normal;
        font-size: clamp(1.6rem, 7vw, 2.15rem);
      }}

      .hero p {{
        white-space: normal;
      }}

      .workspace {{
        grid-template-columns: 1fr;
        gap: 28px;
      }}

      .note {{
        padding: 20px;
      }}

      textarea {{
        min-height: 190px;
        font-size: .9rem;
      }}

      .paper::after {{
        width: 34px;
        height: 34px;
      }}

      .verdict-layout {{
        grid-template-columns: 1fr;
        justify-items: start;
      }}

      .standby-card .card-body {{
        grid-template-columns: 1fr;
      }}

      .stamp {{
        width: 72px;
        height: 72px;
        font-size: .58rem;
      }}

      .dna-grid {{
        grid-template-columns: 1fr;
      }}

      .result-top {{
        grid-template-columns: 1fr;
      }}

      .copy-card {{
        grid-template-columns: 1fr;
      }}

      .copy-btn {{
        min-height: 58px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <img src="{logo_uri}" alt="" />
        <div class="brand-copy">
          <h1>PhishBrake</h1>
          <p class="tagline">Private scam defense for someone you love.</p>
        </div>
      </div>
      <div class="status" aria-label="Runtime status">
        <span>{model_label}</span>
        <span><span id="modelState">READY</span></span>
      </div>
    </header>

    <section class="hero" aria-labelledby="heroTitle">
      <h2 id="heroTitle">Is this message safe?</h2>
      <p>Paste a text, email, or DM. PhishBrake will show the risk, the warning signs, and the safest next step before you reply, click, or pay.</p>
      <div class="safety-note">Safety aid only, not legal, financial, or cybersecurity advice. Always verify through official channels. First check may take up to 30s while the secure system wakes up.</div>
    </section>

    <section class="workspace">
      <aside>
        <form id="scanForm" class="paper note">
          <p class="section-label">Message to check</p>
          <p class="privacy-note">For your privacy: remove passwords, account numbers, ID numbers, addresses, and personal codes before pasting.</p>
          <textarea id="messageInput" placeholder="Paste a text, email, or DM here."></textarea>
          <button id="scanButton" class="run" type="submit">Check message</button>
          <div class="samples" id="samples" aria-label="Sample messages"></div>
        </form>
        <section class="paper memory" aria-live="polite">
          <h3>Earlier checks</h3>
          <div class="memory-list" id="memoryList">No messages checked yet this session.</div>
        </section>
      </aside>

      <section id="result" class="result-stack" aria-live="polite"></section>
    </section>

    <footer class="footer-strip" aria-label="Build details">
      <span><strong>Built for ML Empowerment Build Challenge 3.0</strong></span>
      <span class="footer-dot">•</span>
      <span>Powered by <strong id="modelLabel">{model_label}</strong></span>
      <span class="footer-dot">•</span>
      <span><strong>Gradio Server</strong></span>
      <span class="footer-dot">•</span>
      <span><strong id="runtimeLabel">{backend_label}</strong> runtime</span>
      <span class="footer-dot">•</span>
      <span>trained and evaled on <strong>Modal</strong></span>
    </footer>
  </main>

  <script type="application/json" id="examplesData">{examples_json}</script>
  <script type="module">
    import {{ Client }} from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

    const examples = JSON.parse(document.getElementById("examplesData").textContent);
    const state = {{ memory: [] }};
    const clientPromise = Client.connect(window.location.origin);
    const form = document.getElementById("scanForm");
    const messageInput = document.getElementById("messageInput");
    const scanButton = document.getElementById("scanButton");
    const result = document.getElementById("result");
    const samples = document.getElementById("samples");
    const memoryList = document.getElementById("memoryList");
    const modelLabel = document.getElementById("modelLabel");
    const modelState = document.getElementById("modelState");

    const labels = {{
      dangerous: ["Dangerous", "This looks dangerous"],
      suspicious: ["Suspicious", "This looks suspicious"],
      needs_check: ["Check first", "Verify before acting"],
      safe: ["Looks safe", "No strong scam pattern"]
    }};

    const dnaLabels = {{
      Impersonates: "Who they pretend to be",
      Pressure: "How they pressure you",
      Ask: "What they want",
      Risk: "What could happen"
    }};

    function escapeHtml(value) {{
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }}

    function humanize(value) {{
      return String(value || "Unknown").replaceAll("_", " ").replaceAll("-", " ");
    }}

    function elapsedText(value) {{
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || numeric <= 0) {{
        return "result";
      }}
      return `${{numeric.toFixed(numeric >= 10 ? 0 : 1)}}s`;
    }}

    function showStandby() {{
      result.innerHTML = `
        <article class="card standby-card">
          <div class="card-head"><span>Ready to check</span><span>private safety check</span></div>
          <div class="card-body">
            <div>
              <div class="shield-mark" aria-hidden="true"></div>
              <h3>Paste first. Act after.</h3>
              <p>PhishBrake checks who the message pretends to be, what it asks you to do, and what could go wrong. The first scan may take longer while the private GPU wakes up.</p>
              <ul class="standby-list" aria-label="What PhishBrake checks">
                <li><strong>1</strong><span>Paste the message exactly as you received it.</span></li>
                <li><strong>2</strong><span>Review the verdict and the safest next step before replying.</span></li>
                <li><strong>3</strong><span>Copy the note if you want a trusted person to check it with you.</span></li>
              </ul>
            </div>
            <div class="standby-preview" aria-label="What PhishBrake returns">
              <div class="standby-tile">
                <strong>Risk</strong>
                <span>Clear safe, suspicious, or dangerous verdict.</span>
              </div>
              <div class="standby-tile">
                <strong>Warning signs</strong>
                <span>Who they pretend to be, what they ask, and what could happen.</span>
              </div>
              <div class="standby-tile">
                <strong>Next step</strong>
                <span>Plain advice you can copy to someone you trust.</span>
              </div>
            </div>
          </div>
        </article>
      `;
    }}

    function showLoading() {{
      result.innerHTML = `
        <article class="card loading-card">
          <div class="card-head"><span>Checking message</span><span id="progressText">Starting</span></div>
          <div class="card-body">
            <div class="fold-loader">
              <div class="fold-bar" id="foldBar" style="--progress: 12%"></div>
              <p class="loading-line">> waking the safety check...</p>
              <p class="loading-line">> reading the message...</p>
              <p class="loading-line muted">> checking pressure and impersonation...</p>
              <p class="loading-line muted">> preparing the safest next step...</p>
            </div>
          </div>
        </article>
      `;
      let progress = 12;
      const timer = window.setInterval(() => {{
        progress = Math.min(88, progress + 8);
        const bar = document.getElementById("foldBar");
        const text = document.getElementById("progressText");
        if (!bar || !text) {{
          window.clearInterval(timer);
          return;
        }}
        bar.style.setProperty("--progress", `${{progress}}%`);
        text.textContent = progress < 88 ? `${{progress}}% complete` : "Almost ready";
      }}, 700);
      return timer;
    }}

    function renderMemory(items) {{
      if (!items || !items.length) {{
        memoryList.textContent = "No messages checked yet this session.";
        return;
      }}
      memoryList.innerHTML = items.slice(-4).reverse().map((item) => {{
        const risk = item.risk_level || "needs_check";
        return `
        <div class="memory-row ${{escapeHtml(risk)}}">
          <span>${{escapeHtml(item.summary)}}</span>
          <strong class="memory-badge ${{escapeHtml(risk)}}">${{escapeHtml(risk)}}</strong>
        </div>
      `}}).join("");
    }}

    function renderVerdict(payload) {{
      const analysis = payload.analysis || {{}};
      const risk = analysis.risk_level || "needs_check";
      const label = labels[risk] || labels.needs_check;
      const dna = analysis.scam_dna || {{}};
      const tactics = analysis.tactics || [];
      modelLabel.textContent = payload.model_label || modelLabel.textContent;
      state.memory = payload.memory || state.memory;
      renderMemory(state.memory);

      const dnaHtml = ["Impersonates", "Pressure", "Ask", "Risk"].map((key) => `
        <div class="dna-item">
          <div class="dna-label">${{dnaLabels[key]}}</div>
          <div class="dna-value">${{escapeHtml(humanize(dna[key]))}}</div>
        </div>
      `).join("");

      const tacticsHtml = tactics.length
        ? tactics.map((tag) => `<span class="tag">${{escapeHtml(humanize(tag))}}</span>`).join("")
        : `<span class="tag">none found</span>`;
      const elapsed = elapsedText(payload.elapsed_seconds);

      result.innerHTML = `
        <div class="result-top">
          <article class="card verdict-card ${{escapeHtml(risk)}}">
            <div class="card-head"><span>PhishBrake says</span><span>${{escapeHtml(elapsed)}}</span></div>
            <div class="card-body">
              <div class="verdict-layout">
                <div class="stamp">${{escapeHtml(label[0])}}</div>
                <div>
                  <h3 class="verdict-title">${{escapeHtml(label[1])}}</h3>
                  <p class="summary">${{escapeHtml(analysis.summary)}}</p>
                </div>
              </div>
            </div>
          </article>

          <article class="card">
            <div class="card-head"><span>Warning signs</span><span>how it works</span></div>
            <div class="card-body">
              <div class="dna-grid">${{dnaHtml}}</div>
              <div class="tactics">${{tacticsHtml}}</div>
            </div>
          </article>
        </div>

        <article class="card action-card">
          <div class="card-head"><span>Safest next step</span><span>copy a note for someone you trust</span></div>
          <div class="card-body">
            <p class="action-title">Recommended action</p>
            <p class="action-text">${{escapeHtml(analysis.safest_action)}}</p>
            <div class="copy-card">
              <div class="copy-note" id="copyNote">${{escapeHtml(payload.copy_plan)}}</div>
              <button class="copy-btn" id="copyButton" type="button">Click to copy</button>
            </div>
          </div>
        </article>
      `;

      document.getElementById("copyButton").addEventListener("click", async () => {{
        await navigator.clipboard.writeText(payload.copy_plan || "");
        const button = document.getElementById("copyButton");
        button.textContent = "Copied";
        window.setTimeout(() => button.textContent = "Click to copy", 1200);
      }});
    }}

    function showError(error) {{
      result.innerHTML = `
        <article class="card">
          <div class="card-head"><span>Could not finish</span><span>try again</span></div>
          <div class="card-body"><div class="error">${{escapeHtml(error.message || error)}}</div></div>
        </article>
      `;
    }}

    examples.forEach((example) => {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample";
      button.textContent = example;
      button.addEventListener("click", () => {{
        messageInput.value = example;
        messageInput.focus();
      }});
      samples.appendChild(button);
    }});

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const message = messageInput.value.trim();
      if (!message) {{
        messageInput.focus();
        return;
      }}
      scanButton.disabled = true;
      modelState.textContent = "CHECKING";
      const timer = showLoading();
      try {{
        const client = await clientPromise;
        const response = await client.predict("/analyze", {{
          message,
          memory: state.memory
        }});
        window.clearInterval(timer);
        renderVerdict(response.data?.[0] || response);
        modelState.textContent = "READY";
      }} catch (error) {{
        window.clearInterval(timer);
        modelState.textContent = "READY";
        showError(error);
      }} finally {{
        scanButton.disabled = false;
      }}
    }});

    showStandby();
  </script>
</body>
</html>"""


def kitchen_table_html() -> str:
    examples_json = json.dumps(EXAMPLES).replace("</", "<\\/")
    logo_uri = logo_data_uri()
    safe_qr_demo_uri = qr_demo_data_uri(SAFE_QR_DEMO_PATH)
    phishing_qr_demo_uri = qr_demo_data_uri(PHISHING_QR_DEMO_PATH)
    model_label = escape(model_status_label())
    backend_label = escape(backend_status_label())
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PhishBrake</title>
  <style>
    :root {
      color-scheme: light;
      --cream: #f5ead8;
      --cream-2: #fff7e8;
      --paper: #fffdf6;
      --paper-warm: #fff2d3;
      --ink: #16120d;
      --muted: #6b5a49;
      --line: #24160e;
      --wood-dark: #6f4527;
      --wood: #b6763f;
      --wood-light: #d8a062;
      --linen: #f8ecd4;
      --linen-red: rgba(188, 57, 49, .16);
      --tape: rgba(255, 233, 156, .82);
      --yellow: #f5c400;
      --red: #d94b48;
      --red-soft: #ffe0dd;
      --green: #23794d;
      --green-soft: #dcf6e8;
      --amber: #bd7412;
      --amber-soft: #fff0c8;
      --blue: #2559d9;
      --blue-soft: #e2ecff;
      --plum: #58304f;
      --shadow: 8px 8px 0 rgba(36, 22, 14, .95);
      --soft-shadow: 0 20px 60px rgba(36, 22, 14, .18);
      --font-display: Georgia, "Times New Roman", Times, serif;
      --font-body: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-label: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }

    * { box-sizing: border-box; }

    html {
      min-height: 100%;
      overflow-x: clip;
      background: var(--cream);
    }

    body {
      margin: 0;
      min-height: 100vh;
      overflow-x: clip;
      color: var(--ink);
      font-family: var(--font-body);
      background:
        radial-gradient(circle at 14% 10%, rgba(245, 196, 0, .22), transparent 23rem),
        radial-gradient(circle at 86% 4%, rgba(217, 75, 72, .13), transparent 19rem),
        linear-gradient(90deg, rgba(36,22,14,.04) 1px, transparent 1px),
        var(--cream);
      background-size: auto, auto, 34px 34px, auto;
    }

    button, textarea { font: inherit; }
    button { cursor: pointer; color: inherit; }

    .table-shell {
      width: min(1420px, calc(100% - 32px));
      margin: 0 auto;
      padding: 26px 0 54px;
    }

    .masthead {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 24px;
      align-items: center;
      padding: 0 0 18px;
      border-bottom: 3px solid var(--line);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }

    .brand img {
      width: 54px;
      height: 54px;
      border: 3px solid var(--line);
      border-radius: 12px;
      background: var(--paper);
      box-shadow: 4px 4px 0 var(--line);
    }

    .brand h1 {
      margin: 0;
      font-family: var(--font-display);
      font-size: clamp(2.25rem, 5vw, 4.35rem);
      line-height: .82;
      letter-spacing: 0;
    }

    .brand p {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: clamp(1rem, 1.4vw, 1.25rem);
      line-height: 1.22;
    }

    .tech-badges {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      max-width: 520px;
      font-family: var(--font-label);
      font-size: .72rem;
      font-weight: 900;
      text-transform: uppercase;
    }

    .tech-badges span {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 6px 10px;
      border: 2px solid var(--line);
      border-radius: 999px;
      background: var(--paper);
    }

    .hero-copy {
      display: grid;
      grid-template-columns: minmax(0, .9fr) minmax(280px, .72fr);
      gap: clamp(22px, 4vw, 64px);
      align-items: end;
      padding: 26px 0 28px;
    }

    .hero-copy h2 {
      margin: 0;
      max-width: 13ch;
      font-family: var(--font-display);
      font-size: clamp(2.35rem, 6vw, 5.35rem);
      line-height: .86;
      letter-spacing: 0;
    }

    .hero-copy p {
      margin: 0;
      color: var(--muted);
      font-size: clamp(1rem, 1.35vw, 1.25rem);
      line-height: 1.45;
    }

    .notice-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }

    .notice {
      width: fit-content;
      max-width: 100%;
      padding: 8px 11px;
      border: 2px solid var(--line);
      border-radius: 999px;
      background: var(--paper);
      color: var(--muted);
      font-family: var(--font-label);
      font-size: .72rem;
      font-weight: 900;
      line-height: 1.25;
    }

    .family-rule {
      position: relative;
      margin-top: 18px;
      max-width: 620px;
      border: 2px solid var(--line);
      border-radius: 12px;
      background: var(--paper);
      padding: 14px 16px 14px 54px;
      box-shadow: 5px 5px 0 rgba(36,22,14,.88);
      transform: rotate(.6deg);
    }

    .family-rule::before {
      content: "";
      position: absolute;
      left: 16px;
      top: 17px;
      width: 24px;
      height: 24px;
      border: 2px solid var(--line);
      border-radius: 50%;
      background:
        radial-gradient(circle at center, var(--red) 0 36%, transparent 37%),
        var(--yellow);
    }

    .family-rule strong {
      display: block;
      margin-bottom: 3px;
      font-family: var(--font-label);
      font-size: .72rem;
      text-transform: uppercase;
    }

    .family-rule span {
      color: var(--muted);
      line-height: 1.35;
    }

    .tabletop {
      position: relative;
      display: grid;
      grid-template-columns: minmax(340px, 430px) minmax(0, 1fr);
      gap: clamp(28px, 4vw, 56px);
      align-items: start;
      padding: clamp(18px, 2.5vw, 34px);
      border: 3px solid var(--line);
      border-radius: 18px;
      background:
        linear-gradient(90deg, var(--linen-red) 0 10px, transparent 10px 34px),
        linear-gradient(0deg, var(--linen-red) 0 10px, transparent 10px 34px),
        radial-gradient(ellipse at 17% 16%, rgba(255,255,255,.22), transparent 10rem),
        repeating-linear-gradient(88deg, rgba(255,255,255,.12) 0 3px, transparent 3px 17px),
        linear-gradient(115deg, var(--wood-light), var(--wood) 44%, var(--wood-dark));
      background-size: 68px 68px, 68px 68px, auto, auto, auto;
      box-shadow: var(--soft-shadow);
      overflow: hidden;
      isolation: isolate;
    }

    .tabletop::before {
      content: "";
      position: absolute;
      inset: 18px auto auto 46%;
      width: 150px;
      height: 150px;
      border: 10px solid rgba(80, 43, 24, .18);
      border-radius: 50%;
      transform: rotate(-14deg);
      pointer-events: none;
    }

    .kitchen-prop {
      position: absolute;
      z-index: 0;
      pointer-events: none;
    }

    .recipe-box {
      right: clamp(18px, 3vw, 56px);
      top: clamp(18px, 3vw, 42px);
      width: 178px;
      height: 88px;
      border: 3px solid var(--line);
      border-radius: 10px;
      background: #c84b38;
      box-shadow: 6px 6px 0 rgba(36,22,14,.95);
      transform: rotate(4deg);
      opacity: .92;
    }

    .recipe-box::before,
    .recipe-box::after {
      content: "";
      position: absolute;
      left: 16px;
      right: 16px;
      height: 28px;
      border: 2px solid var(--line);
      border-bottom: 0;
      border-radius: 8px 8px 0 0;
      background: var(--paper);
    }

    .recipe-box::before { top: -24px; transform: rotate(-3deg); }
    .recipe-box::after { top: -14px; transform: rotate(3deg); }

    .tea-cup {
      right: clamp(22px, 4vw, 82px);
      bottom: clamp(22px, 4vw, 72px);
      width: 118px;
      height: 118px;
      border: 4px solid var(--line);
      border-radius: 50%;
      background:
        radial-gradient(circle at center, #6f3825 0 28%, #f4ead7 29% 46%, transparent 47%),
        var(--paper);
      box-shadow: 7px 7px 0 rgba(36,22,14,.9);
      transform: rotate(-8deg);
      opacity: .95;
    }

    .tea-cup::after {
      content: "";
      position: absolute;
      right: -26px;
      top: 34px;
      width: 36px;
      height: 46px;
      border: 4px solid var(--line);
      border-left: 0;
      border-radius: 0 999px 999px 0;
      background: transparent;
    }

    .spoon {
      left: 48%;
      bottom: 20px;
      width: 18px;
      height: 190px;
      border: 3px solid var(--line);
      border-radius: 999px;
      background: linear-gradient(#f7f1df, #bfb5a2);
      box-shadow: 4px 4px 0 rgba(36,22,14,.85);
      transform: rotate(58deg);
      opacity: .52;
    }

    .quilt-runner {
      left: -42px;
      top: 18%;
      width: 180px;
      height: 70%;
      border: 3px solid rgba(36,22,14,.62);
      border-radius: 999px;
      background:
        linear-gradient(45deg, rgba(255,255,255,.22) 25%, transparent 25% 50%, rgba(255,255,255,.22) 50% 75%, transparent 75%),
        linear-gradient(135deg, rgba(217,75,72,.25), rgba(245,196,0,.22));
      background-size: 34px 34px, auto;
      transform: rotate(8deg);
      opacity: .62;
    }

    .tabletop::after {
      content: "";
      position: absolute;
      right: 46px;
      bottom: 34px;
      width: 72px;
      height: 72px;
      border-radius: 50%;
      border: 18px solid rgba(255,255,255,.26);
      background: rgba(255,255,255,.14);
      box-shadow: 0 0 0 3px rgba(36,22,14,.2);
      pointer-events: none;
    }

    .left-stack,
    .right-stack {
      position: relative;
      z-index: 1;
      display: grid;
      gap: 18px;
    }

    .clipboard {
      border: 3px solid var(--line);
      border-radius: 14px;
      background: #f2d18d;
      box-shadow: var(--shadow);
      padding: 16px;
      transform: rotate(-1.2deg);
    }

    .clip {
      width: 92px;
      height: 26px;
      margin: -4px auto 12px;
      border: 3px solid var(--line);
      border-radius: 0 0 14px 14px;
      background: linear-gradient(#c48a36, #8c5b25);
      box-shadow: inset 0 3px 0 rgba(255,255,255,.28);
    }

    .pause-note {
      width: fit-content;
      max-width: 100%;
      margin: 0 auto 14px;
      padding: 9px 13px;
      border: 2px solid var(--line);
      border-radius: 4px;
      background: var(--tape);
      font-family: var(--font-label);
      font-size: .72rem;
      font-weight: 950;
      text-transform: uppercase;
      transform: rotate(1.8deg);
      box-shadow: 3px 3px 0 rgba(36,22,14,.72);
    }

    .paper-card {
      position: relative;
      border: 2.5px solid var(--line);
      border-radius: 10px;
      background:
        linear-gradient(var(--paper) 31px, rgba(36,22,14,.07) 32px),
        var(--paper);
      background-size: 100% 32px;
      padding: 20px;
    }

    .paper-card::after {
      content: "";
      position: absolute;
      right: 18px;
      top: -11px;
      width: 72px;
      height: 23px;
      border: 2px solid rgba(36,22,14,.22);
      background: var(--tape);
      transform: rotate(3deg);
    }

    .paper-card h3,
    .memo h3,
    .history-card h3 {
      margin: 0 0 12px;
      font-family: var(--font-label);
      font-size: .82rem;
      font-weight: 950;
      text-transform: uppercase;
      letter-spacing: .02em;
    }

    .privacy-note {
      margin: 0 0 14px;
      color: var(--muted);
      font-family: var(--font-label);
      font-size: .72rem;
      font-weight: 850;
      line-height: 1.35;
    }

    .qr-upload-label {
      display: block;
      margin: 16px 0 7px;
      color: var(--ink);
      font-family: var(--font-label);
      font-size: .74rem;
      font-weight: 900;
    }

    .qr-image-input {
      display: block;
      width: 100%;
      max-width: 100%;
      color: var(--muted);
      font-size: .8rem;
    }

    .qr-demo-button {
      min-width: 0;
      min-height: 38px;
      margin-top: 8px;
      padding: 7px 12px;
      border: 2px solid var(--line);
      border-radius: 6px;
      background: var(--paper-warm);
      color: var(--ink);
      font-family: var(--font-label);
      font-size: .72rem;
      font-weight: 900;
    }

    .qr-demo-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
    }

    .qr-status {
      margin: 8px 0 0;
      color: #17663b;
      font-family: var(--font-label);
      font-size: .72rem;
      font-weight: 900;
    }

    .qr-status[hidden] {
      display: none;
    }

    textarea {
      width: 100%;
      min-height: 246px;
      resize: vertical;
      border: 2px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(var(--paper) 31px, rgba(36,22,14,.08) 32px),
        var(--paper);
      background-size: 100% 32px;
      padding: 16px;
      color: var(--ink);
      font-size: 1.02rem;
      line-height: 1.55;
      outline: 3px solid transparent;
      outline-offset: 2px;
    }

    textarea:focus-visible {
      outline-color: rgba(245,196,0,.88);
      box-shadow: 0 0 0 5px rgba(245,196,0,.22);
    }

    .check-button {
      position: relative;
      width: 100%;
      min-height: 64px;
      margin-top: 18px;
      border: 3px solid var(--line);
      border-radius: 999px;
      background: var(--yellow);
      color: var(--ink);
      font-family: var(--font-label);
      font-size: 1rem;
      font-weight: 950;
      text-transform: uppercase;
      box-shadow: 0 7px 0 var(--line);
      transition: transform .16s ease, box-shadow .16s ease, opacity .16s ease;
    }

    .check-button::after {
      content: "";
      position: absolute;
      inset: 9px 14px auto auto;
      width: 42px;
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,.35);
    }

    .check-button:active {
      transform: translateY(5px);
      box-shadow: 0 2px 0 var(--line);
    }

    .check-button[disabled] {
      opacity: .62;
      cursor: wait;
    }

    .check-button:focus-visible,
    .sample:focus-visible,
    .copy-button:focus-visible {
      outline: 3px solid var(--blue);
      outline-offset: 3px;
    }

    .samples {
      display: grid;
      gap: 10px;
      margin-top: 20px;
    }

    .sample-label {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 18px 0 -8px;
      color: var(--muted);
      font-family: var(--font-label);
      font-size: .72rem;
      font-weight: 950;
      text-transform: uppercase;
    }

    .sample-label::before {
      content: "";
      width: 16px;
      height: 16px;
      border: 2px solid var(--line);
      border-radius: 4px;
      background: var(--yellow);
      transform: rotate(-8deg);
    }

    .sample {
      border: 2px solid var(--line);
      border-radius: 999px;
      background: var(--paper);
      padding: 10px 14px;
      text-align: left;
      font-size: .92rem;
      line-height: 1.25;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      box-shadow: 3px 3px 0 rgba(36,22,14,.9);
    }

    .history-card {
      border: 3px solid var(--line);
      border-radius: 14px;
      background: var(--paper);
      box-shadow: 5px 5px 0 var(--line);
      padding: 18px;
      transform: rotate(.8deg);
    }

    .trusted-card {
      border: 3px solid var(--line);
      border-radius: 14px;
      background:
        linear-gradient(90deg, rgba(37,89,217,.13) 0 8px, transparent 8px 20px),
        var(--blue-soft);
      box-shadow: 5px 5px 0 var(--line);
      padding: 17px;
      transform: rotate(-.8deg);
    }

    .trusted-card h3 {
      margin: 0 0 8px;
      font-family: var(--font-display);
      font-size: 1.35rem;
      line-height: 1;
    }

    .trusted-card p {
      margin: 0;
      color: var(--muted);
      font-size: .94rem;
      line-height: 1.4;
    }

    .history-list {
      display: grid;
      gap: 10px;
    }

    .history-empty {
      color: var(--muted);
      font-size: .92rem;
    }

    .history-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      border: 2px solid var(--line);
      border-radius: 8px;
      background: var(--red-soft);
      padding: 12px;
      font-size: .82rem;
      line-height: 1.3;
    }

    .history-row.safe { background: var(--green-soft); }
    .history-row.suspicious,
    .history-row.needs_check { background: var(--amber-soft); }

    .history-badge {
      border: 2px solid var(--line);
      border-radius: 999px;
      padding: 5px 8px;
      background: var(--red);
      color: #fff;
      font-family: var(--font-label);
      font-size: .62rem;
      font-weight: 950;
      text-transform: uppercase;
    }

    .history-badge.safe { background: var(--green); }
    .history-badge.suspicious,
    .history-badge.needs_check { background: var(--amber); }

    .memo,
    .result-card {
      position: relative;
      border: 3px solid var(--line);
      border-radius: 16px;
      background: var(--paper);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .memo {
      transform: rotate(.6deg);
    }

    .memo::before,
    .result-card::before,
    .safe-action::before {
      content: "";
      position: absolute;
      left: 50%;
      top: -13px;
      width: 86px;
      height: 25px;
      border: 2px solid rgba(36,22,14,.3);
      background: var(--tape);
      transform: translateX(-50%) rotate(-2deg);
      z-index: 2;
    }

    .memo-header,
    .result-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 48px;
      padding: 12px 18px;
      border-bottom: 3px solid var(--line);
      background: var(--paper-warm);
      font-family: var(--font-label);
      font-weight: 950;
      font-size: .82rem;
      text-transform: uppercase;
    }

    .memo-body,
    .result-body {
      padding: clamp(20px, 3vw, 34px);
    }

    .standby {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 260px;
      gap: clamp(20px, 3vw, 42px);
      align-items: center;
      min-height: 430px;
    }

    .standby h3 {
      margin: 0 0 12px;
      font-family: var(--font-display);
      font-size: clamp(2.2rem, 4vw, 4rem);
      line-height: .9;
      letter-spacing: 0;
    }

    .standby p {
      max-width: 44rem;
      margin: 0;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.55;
    }

    .standby-steps {
      display: grid;
      gap: 12px;
      margin: 24px 0 0;
      padding: 0;
      list-style: none;
    }

    .standby-steps li {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      max-width: 46rem;
      font-size: 1rem;
      line-height: 1.35;
    }

    .standby-steps strong {
      display: inline-grid;
      place-items: center;
      width: 1.7rem;
      height: 1.7rem;
      border: 2px solid var(--line);
      border-radius: 999px;
      background: var(--yellow);
      font-family: var(--font-label);
      font-size: .86rem;
    }

    .coaster-stack {
      display: grid;
      gap: 14px;
    }

    .coaster {
      min-height: 104px;
      display: grid;
      align-content: center;
      gap: 7px;
      border: 3px solid var(--line);
      border-radius: 12px;
      background: var(--cream-2);
      padding: 16px;
      box-shadow: 4px 4px 0 var(--line);
    }

    .coaster strong {
      font-family: var(--font-label);
      font-size: .76rem;
      text-transform: uppercase;
    }

    .coaster span {
      color: var(--muted);
      line-height: 1.35;
    }

    .fridge-note {
      position: relative;
      border: 3px solid var(--line);
      border-radius: 18px;
      background:
        linear-gradient(var(--paper) 31px, rgba(36,22,14,.08) 32px),
        var(--paper);
      background-size: 100% 32px;
      padding: 18px;
      box-shadow: 5px 5px 0 var(--line);
      transform: rotate(1.8deg);
    }

    .fridge-note::before {
      content: "";
      position: absolute;
      left: 50%;
      top: -16px;
      width: 30px;
      height: 30px;
      border: 3px solid var(--line);
      border-radius: 50%;
      background: var(--red);
      transform: translateX(-50%);
      box-shadow: inset 0 0 0 6px rgba(255,255,255,.28);
    }

    .fridge-note h4 {
      margin: 10px 0 10px;
      font-family: var(--font-display);
      font-size: 1.6rem;
      line-height: .95;
    }

    .fridge-note ul {
      display: grid;
      gap: 9px;
      margin: 0;
      padding: 0;
      list-style: none;
      color: var(--muted);
      line-height: 1.28;
    }

    .fridge-note li::before {
      content: "STOP";
      display: inline-block;
      margin-right: 8px;
      border: 2px solid var(--line);
      border-radius: 999px;
      background: var(--red);
      color: #fff;
      padding: 2px 5px;
      font-family: var(--font-label);
      font-size: .58rem;
      font-weight: 950;
    }

    .loading .result-body {
      min-height: 410px;
      display: grid;
      align-content: center;
      gap: 20px;
      background:
        radial-gradient(circle at 80% 20%, rgba(245,196,0,.16), transparent 16rem),
        var(--paper);
    }

    .reader-light {
      height: 22px;
      border: 3px solid var(--line);
      border-radius: 999px;
      background:
        linear-gradient(90deg, var(--yellow) var(--progress, 12%), transparent 0),
        repeating-linear-gradient(90deg, rgba(36,22,14,.16) 0 8px, transparent 8px 16px),
        var(--paper);
      box-shadow: 5px 5px 0 var(--line);
    }

    .scan-note {
      border: 3px solid var(--line);
      border-radius: 14px;
      background: var(--cream-2);
      padding: 22px;
      transform: rotate(-.7deg);
      box-shadow: 5px 5px 0 var(--line);
    }

    .scan-note::before {
      content: "";
      display: block;
      width: 64px;
      height: 64px;
      margin: 0 auto 14px;
      border: 3px solid var(--line);
      border-radius: 50%;
      background:
        radial-gradient(circle at center, var(--red) 0 25%, transparent 26%),
        radial-gradient(circle at center, transparent 0 42%, var(--yellow) 43% 60%, transparent 61%),
        var(--paper);
      animation: carefulPulse 1.2s ease-in-out infinite alternate;
    }

    @keyframes carefulPulse {
      from { transform: scale(.95) rotate(-2deg); }
      to { transform: scale(1.05) rotate(2deg); }
    }

    .scan-note h3 {
      margin: 0 0 14px;
      font-family: var(--font-display);
      font-size: clamp(1.8rem, 3vw, 2.8rem);
      line-height: .95;
    }

    .scan-lines {
      display: grid;
      gap: 10px;
      margin: 18px 0 0;
      padding: 0;
      list-style: none;
      font-family: var(--font-label);
      font-weight: 900;
      color: var(--muted);
    }

    .scan-lines li.active {
      color: var(--ink);
    }

    .result-grid {
      display: grid;
      grid-template-columns: minmax(255px, .62fr) minmax(0, 1fr);
      gap: 18px;
      align-items: stretch;
    }

    .decision-strip {
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 4px;
    }

    .decision-light {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 7px 10px;
      border: 2px solid var(--line);
      border-radius: 999px;
      background: var(--paper);
      color: var(--muted);
      font-family: var(--font-label);
      font-size: .66rem;
      font-weight: 950;
      text-transform: uppercase;
    }

    .decision-light::before {
      content: "";
      width: 10px;
      height: 10px;
      border: 2px solid var(--line);
      border-radius: 50%;
      background: #d7c7a6;
    }

    .decision-light.on.dangerous::before { background: var(--red); }
    .decision-light.on.needs_check::before,
    .decision-light.on.suspicious::before { background: var(--amber); }
    .decision-light.on.safe::before { background: var(--green); }

    .verdict-panel {
      position: relative;
      min-height: 285px;
      display: grid;
      align-content: center;
      justify-items: center;
      gap: 18px;
      border: 3px solid var(--line);
      border-radius: 14px;
      background: var(--red-soft);
      padding: 26px;
      text-align: center;
    }

    .verdict-panel::before {
      content: "";
      position: absolute;
      left: -3px;
      top: 24px;
      bottom: 24px;
      width: 16px;
      border: 3px solid var(--line);
      border-left: 0;
      border-radius: 0 999px 999px 0;
      background: var(--red);
    }

    .verdict-panel.safe::before { background: var(--green); }
    .verdict-panel.suspicious::before,
    .verdict-panel.needs_check::before { background: var(--amber); }

    .verdict-panel.safe { background: var(--green-soft); }
    .verdict-panel.suspicious,
    .verdict-panel.needs_check { background: var(--amber-soft); }

    .rubber-stamp {
      width: min(180px, 52vw);
      aspect-ratio: 1;
      display: grid;
      place-items: center;
      border: 7px double currentColor;
      border-radius: 999px;
      color: var(--red);
      font-family: var(--font-label);
      font-weight: 950;
      font-size: clamp(.84rem, 1.4vw, 1.15rem);
      text-align: center;
      text-transform: uppercase;
      transform: rotate(-11deg);
      animation: stampIn .34s cubic-bezier(.2, .95, .3, 1.2);
    }

    .safe .rubber-stamp { color: var(--green); }
    .suspicious .rubber-stamp,
    .needs_check .rubber-stamp { color: var(--amber); }

    @keyframes stampIn {
      0% { transform: scale(1.55) rotate(-11deg); opacity: 0; filter: blur(3px); }
      70% { transform: scale(.92) rotate(-11deg); opacity: 1; filter: blur(0); }
      100% { transform: scale(1) rotate(-11deg); }
    }

    .verdict-panel h3 {
      margin: 0;
      font-family: var(--font-display);
      font-size: clamp(1.75rem, 3vw, 3rem);
      line-height: .94;
    }

    .verdict-panel p {
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }

    .confidence-card {
      width: 100%;
      display: grid;
      gap: 3px;
      border: 2px solid var(--line);
      border-radius: 999px;
      background: var(--paper);
      padding: 9px 13px;
      text-align: left;
    }

    .confidence-card strong {
      font-family: var(--font-label);
      font-size: .66rem;
      font-weight: 950;
      text-transform: uppercase;
    }

    .confidence-card span {
      color: var(--muted);
      font-size: .82rem;
      line-height: 1.25;
    }

    .dna-board {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }

    .dna-card {
      min-height: 122px;
      border: 3px solid var(--line);
      border-radius: 12px;
      background:
        linear-gradient(135deg, var(--paper) 0 78%, #eedbb7 79%),
        var(--paper);
      padding: 14px;
      box-shadow: 3px 3px 0 var(--line);
    }

    .dna-card small {
      display: block;
      margin-bottom: 12px;
      font-family: var(--font-label);
      font-size: .68rem;
      font-weight: 950;
      text-transform: uppercase;
    }

    .dna-card strong {
      display: block;
      font-family: var(--font-display);
      font-size: clamp(1rem, 1.5vw, 1.35rem);
      line-height: 1.05;
      overflow-wrap: anywhere;
    }

    .tactic-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }

    .dna-caption {
      margin: 0 0 12px;
      font-family: var(--font-label);
      font-size: .72rem;
      font-weight: 950;
      text-transform: uppercase;
      color: var(--muted);
    }

    .tag {
      border: 2px solid var(--line);
      border-radius: 999px;
      background: var(--red);
      color: #fff;
      padding: 6px 9px;
      font-family: var(--font-label);
      font-size: .68rem;
      font-weight: 950;
      text-transform: uppercase;
    }

    .safe-action {
      position: relative;
      margin-top: 18px;
      border: 3px solid var(--line);
      border-radius: 16px;
      background: var(--green-soft);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .safe-action::after {
      content: "";
      position: absolute;
      right: 18px;
      top: 56px;
      width: 90px;
      height: 90px;
      border: 3px solid rgba(36,22,14,.18);
      border-radius: 50%;
      background:
        radial-gradient(circle at center, rgba(35,121,77,.3) 0 30%, transparent 31%),
        transparent;
      transform: rotate(12deg);
      pointer-events: none;
    }

    .safe-action-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 12px 18px;
      border-bottom: 3px solid var(--line);
      background: var(--paper-warm);
      font-family: var(--font-label);
      font-size: .82rem;
      font-weight: 950;
      text-transform: uppercase;
    }

    .safe-action-body {
      padding: clamp(18px, 3vw, 28px);
    }

    .safe-action-body h3 {
      margin: 0 0 12px;
      font-family: var(--font-label);
      font-size: .78rem;
      text-transform: uppercase;
    }

    .safe-action-body .action-text {
      margin: 0 0 20px;
      font-family: var(--font-display);
      font-size: clamp(1.3rem, 2.4vw, 2.35rem);
      font-weight: 850;
      line-height: 1.02;
    }

    .copy-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 170px;
      gap: 14px;
      align-items: stretch;
    }

    .copy-note {
      max-height: 170px;
      overflow: auto;
      white-space: pre-wrap;
      border: 3px solid var(--line);
      border-radius: 10px;
      background: var(--paper);
      padding: 14px;
      font-size: .88rem;
      line-height: 1.4;
    }

    .copy-button {
      border: 3px solid var(--line);
      border-radius: 10px;
      background: var(--paper);
      font-family: var(--font-label);
      font-size: .86rem;
      font-weight: 950;
      text-transform: uppercase;
      box-shadow: 5px 5px 0 var(--line);
      transition: transform .16s ease, box-shadow .16s ease;
    }

    .copy-button::before {
      content: "Shareable note";
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: .62rem;
      font-weight: 950;
    }

    .copy-button:active {
      transform: translateY(4px);
      box-shadow: 1px 1px 0 var(--line);
    }

    .footer-strip {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 8px 12px;
      margin-top: 26px;
      color: rgba(255,255,255,.92);
      font-family: var(--font-label);
      font-size: .78rem;
      font-weight: 850;
      text-align: center;
      text-shadow: 0 1px 0 rgba(0,0,0,.35);
    }

    .footer-strip strong { color: #fff; }

    .error {
      border: 3px solid var(--line);
      border-radius: 12px;
      background: var(--red-soft);
      padding: 16px;
      font-weight: 850;
    }

    @media (max-width: 1120px) {
      .masthead,
      .hero-copy,
      .tabletop,
      .standby,
      .result-grid {
        grid-template-columns: 1fr;
      }

      .tech-badges {
        justify-content: flex-start;
      }
    }

    @media (max-width: 720px) {
      .table-shell {
        width: min(100% - 20px, 620px);
        padding-top: 16px;
      }

      .brand img {
        width: 44px;
        height: 44px;
      }

      .hero-copy h2 {
        max-width: none;
        font-size: clamp(2.25rem, 13vw, 3.65rem);
      }

      .tabletop {
        padding: 14px;
        border-radius: 14px;
      }

      .clipboard,
      .history-card,
      .memo,
      .trusted-card,
      .fridge-note {
        transform: none;
      }

      .kitchen-prop {
        display: none;
      }

      .paper-card,
      .memo-body,
      .result-body {
        padding: 16px;
      }

      textarea {
        min-height: 190px;
        font-size: .96rem;
      }

      .dna-board,
      .copy-grid {
        grid-template-columns: 1fr;
      }

      .copy-button {
        min-height: 66px;
      }

      .family-rule {
        padding-left: 16px;
      }

      .family-rule::before {
        display: none;
      }
    }

    @media (max-width: 520px) {
      body {
        background:
          radial-gradient(circle at 12% 2%, rgba(245, 196, 0, .2), transparent 14rem),
          var(--cream);
      }

      .table-shell {
        width: calc(100% - 14px);
        padding: 10px 0 28px;
      }

      .masthead {
        gap: 12px;
        padding-bottom: 12px;
      }

      .brand {
        align-items: flex-start;
        gap: 10px;
      }

      .brand img {
        width: 40px;
        height: 40px;
        border-width: 2px;
        border-radius: 10px;
        box-shadow: 3px 3px 0 var(--line);
        flex: 0 0 auto;
      }

      .brand h1 {
        font-size: clamp(2.25rem, 12vw, 3rem);
        line-height: .86;
      }

      .brand p {
        margin-top: 6px;
        font-size: .95rem;
      }

      .tech-badges {
        width: 100%;
        gap: 7px;
        font-size: .64rem;
      }

      .tech-badges span {
        min-height: 30px;
        max-width: 100%;
        justify-content: center;
        text-align: center;
        white-space: normal;
        line-height: 1.15;
      }

      .hero-copy {
        gap: 14px;
        padding: 18px 0 18px;
      }

      .hero-copy h2 {
        font-size: clamp(2.05rem, 11vw, 3.15rem);
        line-height: .9;
      }

      .hero-copy p {
        font-size: .98rem;
        line-height: 1.38;
      }

      .family-rule {
        margin-top: 14px;
        padding: 13px 14px;
        border-radius: 10px;
        transform: none;
        box-shadow: 4px 4px 0 rgba(36,22,14,.88);
      }

      .notice-row {
        display: grid;
        gap: 8px;
      }

      .notice {
        width: 100%;
        border-radius: 12px;
        font-size: .66rem;
      }

      .tabletop {
        padding: 10px;
        border-width: 2px;
        border-radius: 13px;
        gap: 18px;
      }

      .clipboard,
      .history-card,
      .trusted-card,
      .memo,
      .result-card,
      .safe-action {
        border-width: 2px;
        border-radius: 12px;
        box-shadow: 4px 4px 0 var(--line);
      }

      .clipboard {
        padding: 10px;
      }

      .clip,
      .pause-note,
      .paper-card::after,
      .memo::before,
      .result-card::before,
      .safe-action::before,
      .safe-action::after {
        display: none;
      }

      .paper-card,
      .memo-body,
      .result-body,
      .safe-action-body {
        padding: 14px;
      }

      .paper-card h3,
      .memo h3,
      .history-card h3 {
        font-size: .76rem;
      }

      .privacy-note {
        font-size: .68rem;
      }

      textarea {
        min-height: 178px;
        padding: 13px;
        font-size: 1rem;
        line-height: 1.45;
      }

      .check-button {
        min-height: 58px;
        font-size: .9rem;
      }

      .check-button::after {
        width: 30px;
      }

      .sample {
        min-height: 44px;
        padding: 10px 12px;
        font-size: .86rem;
      }

      .trusted-card,
      .history-card {
        padding: 14px;
      }

      .trusted-card h3 {
        font-size: 1.18rem;
      }

      .trusted-card p,
      .history-empty {
        font-size: .88rem;
      }

      .memo-header,
      .result-header,
      .safe-action-header {
        min-height: 42px;
        padding: 10px 12px;
        gap: 10px;
        font-size: .68rem;
        line-height: 1.15;
      }

      .standby {
        min-height: 0;
      }

      .standby h3 {
        font-size: clamp(2.1rem, 11vw, 3rem);
      }

      .standby p,
      .standby-steps li {
        font-size: .95rem;
      }

      .fridge-note {
        padding: 14px;
      }

      .loading .result-body {
        min-height: 260px;
      }

      .scan-note {
        padding: 16px;
        transform: none;
        box-shadow: 4px 4px 0 var(--line);
      }

      .scan-note::before {
        width: 48px;
        height: 48px;
      }

      .scan-note h3 {
        font-size: clamp(1.8rem, 9vw, 2.45rem);
      }

      .scan-lines {
        font-size: .78rem;
        line-height: 1.35;
      }

      .result-grid {
        gap: 14px;
      }

      .verdict-panel {
        min-height: 0;
        padding: 18px 14px;
        gap: 13px;
      }

      .verdict-panel::before {
        width: 10px;
        top: 18px;
        bottom: 18px;
      }

      .rubber-stamp {
        width: min(140px, 44vw);
        border-width: 5px;
      }

      .verdict-panel h3 {
        font-size: clamp(1.75rem, 9vw, 2.55rem);
      }

      .verdict-panel p,
      .copy-note {
        font-size: .9rem;
      }

      .decision-strip {
        gap: 6px;
      }

      .decision-light {
        padding: 6px 8px;
        font-size: .58rem;
      }

      .dna-board {
        gap: 10px;
      }

      .dna-card {
        min-height: 0;
        padding: 12px;
        border-width: 2px;
        box-shadow: 2px 2px 0 var(--line);
      }

      .dna-card small,
      .dna-caption,
      .tag {
        font-size: .62rem;
      }

      .dna-card strong {
        font-size: clamp(1rem, 5vw, 1.32rem);
      }

      .safe-action-body .action-text {
        font-size: clamp(1.45rem, 8vw, 2rem);
        line-height: 1.03;
      }

      .copy-note {
        max-height: 210px;
      }

      .copy-button {
        min-height: 58px;
      }

      .history-row {
        grid-template-columns: 1fr;
        gap: 8px;
        font-size: .78rem;
      }

      .history-badge {
        width: fit-content;
      }

      .footer-strip {
        margin-top: 18px;
        color: rgba(255,255,255,.96);
        font-size: .66rem;
      }
    }
  </style>
</head>
<body>
  <main class="table-shell">
    <header class="masthead">
      <div class="brand">
        <img src="__LOGO_URI__" alt="" />
        <div>
          <h1>PhishBrake</h1>
          <p>When something feels off, check it here.</p>
        </div>
      </div>
      <div class="tech-badges" aria-label="Build details">
        <span>__MODEL_LABEL__</span>
        <span id="modelState">Ready</span>
      </div>
    </header>

    <section class="hero-copy" aria-labelledby="heroTitle">
      <div>
        <h2 id="heroTitle">Pull up a chair. Let's read it together.</h2>
      </div>
      <div>
        <p>Paste a text, email, or DM that feels off. PhishBrake turns it into a simple safety note: the risk, the warning signs, and the safest next step before you click, reply, or pay.</p>
        <div class="family-rule">
          <strong>The kitchen-table rule</strong>
          <span>If a message creates panic, asks for money, or sends you to a link, bring it here before you act.</span>
        </div>
        <div class="notice-row">
          <div class="notice">Safety aid only. Always verify through official channels.</div>
          <div class="notice">First check may take up to 30s while the secure system wakes up.</div>
        </div>
      </div>
    </section>

    <section class="tabletop">
      <div class="kitchen-prop recipe-box" aria-hidden="true"></div>
      <div class="kitchen-prop tea-cup" aria-hidden="true"></div>
      <div class="kitchen-prop spoon" aria-hidden="true"></div>
      <div class="kitchen-prop quilt-runner" aria-hidden="true"></div>
      <aside class="left-stack">
        <form id="scanForm" class="clipboard">
          <div class="clip" aria-hidden="true"></div>
          <div class="pause-note">If it asks for money, pause.</div>
          <div class="paper-card">
            <h3>Message to check</h3>
            <p class="privacy-note">For your privacy: remove passwords, account numbers, ID numbers, addresses, and personal codes before pasting.</p>
            <textarea id="messageInput" placeholder="Paste the message here exactly as you received it."></textarea>
            <label class="qr-upload-label" for="qrImageInput">Upload suspicious screenshot or QR code (optional)</label>
            <input class="qr-image-input" id="qrImageInput" type="file" accept="image/*">
            <div class="qr-demo-row" aria-label="QR code demo choices">
              <button class="qr-demo-button" id="safeQrDemoButton" type="button">Load safe QR demo</button>
              <button class="qr-demo-button" id="phishingQrDemoButton" type="button">Load phishing QR demo</button>
            </div>
            <p class="qr-status" id="qrStatus" role="status" aria-live="polite" hidden></p>
            <button id="scanButton" class="check-button" type="submit">Check this message</button>
            <div class="sample-label">Try homograph or cache demos; repeat the cache sample for an instant hit</div>
            <div class="samples" id="samples" aria-label="Sample messages"></div>
          </div>
        </form>

        <section class="trusted-card">
          <h3>Call the number you already trust.</h3>
          <p>If PhishBrake says stop or check, do not use the number or link in the message. Open the official app, website, or a saved contact.</p>
        </section>

        <section class="history-card" aria-live="polite">
          <h3>Earlier checks</h3>
          <div class="history-list" id="memoryList"><span class="history-empty">No messages checked yet this session.</span></div>
        </section>
      </aside>

      <section class="right-stack" id="result" aria-live="polite"></section>

      <footer class="footer-strip" aria-label="Runtime details">
        <span>Built for <strong>ML Empowerment Build Challenge 3.0</strong></span>
        <span>Powered by <strong id="modelLabel">__MODEL_LABEL__</strong></span>
        <span><strong>Gradio Server</strong></span>
        <span><strong>__BACKEND_LABEL__</strong> runtime</span>
      </footer>
    </section>
  </main>

  <script type="application/json" id="examplesData">__EXAMPLES_JSON__</script>
  <script type="module">
    import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

    const examples = JSON.parse(document.getElementById("examplesData").textContent);
    const state = { memory: [] };
    const clientPromise = Client.connect(window.location.origin);
    const form = document.getElementById("scanForm");
    const messageInput = document.getElementById("messageInput");
    const qrImageInput = document.getElementById("qrImageInput");
    const safeQrDemoButton = document.getElementById("safeQrDemoButton");
    const phishingQrDemoButton = document.getElementById("phishingQrDemoButton");
    const qrStatus = document.getElementById("qrStatus");
    const qrDemoImages = {
      safe: "__SAFE_QR_DEMO_DATA__",
      phishing: "__PHISHING_QR_DEMO_DATA__"
    };
    const scanButton = document.getElementById("scanButton");
    const result = document.getElementById("result");
    const samples = document.getElementById("samples");
    const memoryList = document.getElementById("memoryList");
    const modelLabel = document.getElementById("modelLabel");
    const modelState = document.getElementById("modelState");

    async function loadQrDemo(kind) {
      const imageData = qrDemoImages[kind];
      if (!imageData) return;
      try {
        const blob = await (await fetch(imageData)).blob();
        const transfer = new DataTransfer();
        transfer.items.add(new File([blob], `jawbreaker-${kind}-qr-demo.png`, { type: blob.type }));
        qrImageInput.files = transfer.files;
        qrStatus.textContent = `${kind === "safe" ? "Safe" : "Phishing"} QR demo image ready to scan.`;
        qrStatus.hidden = false;
      } catch (error) {
        qrStatus.textContent = "Could not load the QR demo image.";
        qrStatus.hidden = false;
      }
    }

    safeQrDemoButton.addEventListener("click", () => loadQrDemo("safe"));
    phishingQrDemoButton.addEventListener("click", () => loadQrDemo("phishing"));

    const labels = {
      dangerous: { stamp: "Do not click", title: "This looks dangerous", className: "dangerous" },
      suspicious: { stamp: "Check first", title: "This needs a closer look", className: "suspicious" },
      needs_check: { stamp: "Check first", title: "Verify before acting", className: "needs_check" },
      safe: { stamp: "Looks ok", title: "No strong scam pattern", className: "safe" }
    };

    const dnaLabels = {
      Impersonates: "Who they pretend to be",
      Pressure: "How they pressure you",
      Ask: "What they want",
      Risk: "What could happen"
    };

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function humanize(value) {
      return String(value || "Unknown").replaceAll("_", " ").replaceAll("-", " ");
    }

    function elapsedText(value) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || numeric <= 0) return "checked";
      return `${numeric.toFixed(numeric >= 10 ? 0 : 1)}s`;
    }

    function readFileDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : null);
        reader.onerror = () => reject(reader.error || new Error("Could not read the image."));
        reader.readAsDataURL(file);
      });
    }

    function showStandby() {
      result.innerHTML = `
        <article class="memo">
          <div class="memo-header"><span>Kitchen table check</span><span>private safety check</span></div>
          <div class="memo-body standby">
            <div>
              <h3>Paste first. Act after.</h3>
              <p>Scammers count on panic. This table is the pause: one message, one check, one safe next step you can share with someone you trust.</p>
              <ol class="standby-steps">
                <li><strong>1</strong><span>Paste the message exactly as you received it.</span></li>
                <li><strong>2</strong><span>Read the stamped verdict and warning signs.</span></li>
                <li><strong>3</strong><span>Copy the note if you want a trusted person to check it with you.</span></li>
              </ol>
            </div>
            <div class="coaster-stack" aria-label="What PhishBrake returns">
              <div class="fridge-note">
                <h4>Before you tap</h4>
                <ul>
                  <li>Do not rush.</li>
                  <li>Do not send codes.</li>
                  <li>Do not pay inside a text.</li>
                </ul>
              </div>
            </div>
          </div>
        </article>
      `;
    }

    function showLoading() {
      result.innerHTML = `
        <article class="result-card loading">
          <div class="result-header"><span>Checking the message</span><span id="progressText">Starting</span></div>
          <div class="result-body">
            <div class="scan-note">
              <h3>Reading this carefully.</h3>
              <div class="reader-light" id="readerLight" style="--progress: 12%"></div>
              <ul class="scan-lines">
                <li class="active">pulling the message closer...</li>
                <li>checking who they pretend to be...</li>
                <li>looking for pressure or payment asks...</li>
                <li>writing the safest next step...</li>
              </ul>
            </div>
          </div>
        </article>
      `;
      let progress = 12;
      const timer = window.setInterval(() => {
        progress = Math.min(88, progress + 8);
        const light = document.getElementById("readerLight");
        const text = document.getElementById("progressText");
        const lines = [...document.querySelectorAll(".scan-lines li")];
        if (!light || !text) {
          window.clearInterval(timer);
          return;
        }
        light.style.setProperty("--progress", `${progress}%`);
        text.textContent = progress < 88 ? `${progress}%` : "Almost ready";
        const activeIndex = Math.min(lines.length - 1, Math.floor(progress / 24));
        lines.forEach((line, index) => line.classList.toggle("active", index <= activeIndex));
      }, 700);
      return timer;
    }

    function renderMemory(items) {
      if (!items || !items.length) {
        memoryList.innerHTML = `<span class="history-empty">No messages checked yet this session.</span>`;
        return;
      }
      memoryList.innerHTML = items.slice(-4).reverse().map((item) => {
        const risk = item.risk_level || "needs_check";
        return `
          <div class="history-row ${escapeHtml(risk)}">
            <span>${escapeHtml(item.summary)}</span>
            <strong class="history-badge ${escapeHtml(risk)}">${escapeHtml(humanize(risk))}</strong>
          </div>
        `;
      }).join("");
    }

    function renderVerdict(payload) {
      const analysis = payload.analysis || {};
      const confidence = payload.confidence || {};
      const risk = analysis.risk_level || "needs_check";
      const label = labels[risk] || labels.needs_check;
      const dna = analysis.scam_dna || {};
      const tactics = analysis.tactics || [];
      modelLabel.textContent = payload.model_label || modelLabel.textContent;
      state.memory = payload.memory || state.memory;
      renderMemory(state.memory);

      const dnaHtml = ["Impersonates", "Pressure", "Ask", "Risk"].map((key) => `
        <div class="dna-card">
          <small>${dnaLabels[key]}</small>
          <strong>${escapeHtml(humanize(dna[key]))}</strong>
        </div>
      `).join("");

      const tacticsHtml = tactics.length
        ? tactics.map((tag) => `<span class="tag">${escapeHtml(humanize(tag))}</span>`).join("")
        : `<span class="tag">none found</span>`;

      result.innerHTML = `
        <article class="result-card">
          <div class="result-header"><span>PhishBrake verdict</span><span>${escapeHtml(elapsedText(payload.elapsed_seconds))}</span></div>
          <div class="result-body">
            <div class="result-grid">
              <section class="verdict-panel ${escapeHtml(label.className)}">
                <div class="decision-strip" aria-label="Safety signal">
                  <span class="decision-light dangerous ${risk === "dangerous" ? "on" : ""}">Stop</span>
                  <span class="decision-light needs_check ${risk === "needs_check" || risk === "suspicious" ? "on" : ""}">Check</span>
                  <span class="decision-light safe ${risk === "safe" ? "on" : ""}">Safe</span>
                </div>
                <div class="rubber-stamp">${escapeHtml(label.stamp)}</div>
                <h3>${escapeHtml(label.title)}</h3>
                <p>${escapeHtml(analysis.summary)}</p>
                <div class="confidence-card">
                  <strong>${escapeHtml(confidence.label || "Context check")}</strong>
                  <span>${escapeHtml(confidence.detail || "Verify before you act.")}</span>
                </div>
              </section>
              <section>
                <p class="dna-caption">Why PhishBrake thinks so</p>
                <div class="dna-board">${dnaHtml}</div>
                <div class="tactic-row">${tacticsHtml}</div>
              </section>
            </div>
          </div>
        </article>

        <article class="safe-action">
          <div class="safe-action-header"><span>What to do before you act</span><span>copy a note for someone you trust</span></div>
          <div class="safe-action-body">
            <h3>Safest next step</h3>
            <p class="action-text">${escapeHtml(analysis.safest_action)}</p>
            <div class="copy-grid">
              <div class="copy-note">${escapeHtml(payload.copy_plan)}</div>
              <button class="copy-button" id="copyButton" type="button">Copy note</button>
            </div>
          </div>
        </article>
      `;

      document.getElementById("copyButton").addEventListener("click", async () => {
        await navigator.clipboard.writeText(payload.copy_plan || "");
        const button = document.getElementById("copyButton");
        button.textContent = "Copied";
        button.style.transform = "rotate(-4deg) scale(1.02)";
        window.setTimeout(() => {
          button.textContent = "Copy note";
          button.style.transform = "";
        }, 1200);
      });
    }

    function showError(error) {
      result.innerHTML = `
        <article class="result-card">
          <div class="result-header"><span>Could not finish</span><span>try again</span></div>
          <div class="result-body"><div class="error">${escapeHtml(error.message || error)}</div></div>
        </article>
      `;
    }

    examples.forEach((example, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample";
      button.textContent = example;
      button.addEventListener("click", () => {
        messageInput.value = example;
        if (index === 0) {
          scanButton.focus();
        } else {
          messageInput.focus();
        }
      });
      samples.appendChild(button);
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = messageInput.value.trim();
      const imageFile = qrImageInput.files?.[0];
      if (!message && !imageFile) {
        messageInput.focus();
        return;
      }
      qrStatus.hidden = true;
      qrStatus.textContent = "";
      scanButton.disabled = true;
      scanButton.textContent = "Checking";
      modelState.textContent = "Checking";
      const timer = showLoading();
      let statusAfterCheck = "Ready";
      try {
        const client = await clientPromise;
        const request = { message, memory: state.memory };
        if (imageFile) request.image_data = await readFileDataUrl(imageFile);
        const response = await client.predict("/analyze", request);
        window.clearInterval(timer);
        const payload = response.data?.[0] || response;
        statusAfterCheck = payload.cache_status || "Ready";
        if (payload.qr_text) {
          qrStatus.textContent = "QR Code Decoded";
          qrStatus.hidden = false;
        }
        renderVerdict(payload);
      } catch (error) {
        window.clearInterval(timer);
        showError(error);
      } finally {
        scanButton.disabled = false;
        scanButton.textContent = "Check this message";
        modelState.textContent = statusAfterCheck;
      }
    });

    showStandby();
  </script>
</body>
</html>"""
    return (
        template.replace("__LOGO_URI__", logo_uri)
        .replace("__MODEL_LABEL__", model_label)
        .replace("__BACKEND_LABEL__", backend_label)
        .replace("__EXAMPLES_JSON__", examples_json)
        .replace("__SAFE_QR_DEMO_DATA__", safe_qr_demo_uri)
        .replace("__PHISHING_QR_DEMO_DATA__", phishing_qr_demo_uri)
    )


def build_server() -> gr.Server:
    app = gr.Server()

    @app.api(name="analyze")
    @gpu_callback
    def analyze_api(
      message: str,
      memory: list[dict] | None = None,
      image_data: str | None = None,
    ) -> dict[str, Any]:
      message, qr_text = _message_with_qr_payload(message, _image_from_data_url(image_data))
      payload = analysis_payload(message, memory)
      if qr_text:
        payload["qr_text"] = qr_text
      return payload

    @app.api(name="warm")
    @gpu_callback
    def warm_api() -> dict[str, str]:
        warm_model_impl()
        return {"status": "ready", "backend": current_backend(), "model": model_status_label()}

    @app.get("/", response_class=HTMLResponse)
    async def homepage() -> HTMLResponse:
        return HTMLResponse(kitchen_table_html())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "backend": current_backend(), "model": model_status_label()}

    return app


def build_handoff_message(message: str, analysis: ScamAnalysis) -> str:
    return build_trusted_note(message, analysis)


def should_use_heuristic_guard(
    model_analysis: ScamAnalysis,
    heuristic: ScamAnalysis,
    validation_errors: list[str],
    message: str = "",
) -> bool:
    return should_apply_heuristic_guard(message, model_analysis, heuristic, validation_errors)


@gpu_callback
def analyze_message(
    message: str,
    memory: list[dict] | None,
    qr_image: Any | None = None,
):
    memory = memory or []
    message, qr_text = _message_with_qr_payload(message, qr_image)
    qr_status = gr.update(value="QR Code Decoded" if qr_text else "", visible=bool(qr_text))
    cached_analysis = _cached_analysis(message) if message.strip() else None
    if cached_analysis is not None:
        entry = memory_entry_from_analysis(message, cached_analysis)
        if not memory or memory[-1].get("text") != entry.get("text"):
            memory.append(entry)
        last_scan = {"message": message, "entry": entry}
        yield (
            f"<p>{CACHE_HIT_STATUS}</p>" + render_analysis_html(message, cached_analysis),
            render_memory_html(cached_analysis, memory),
            memory,
            last_scan,
            gr.update(interactive=True),
            gr.update(interactive=True),
            qr_status,
        )
        return
    if not message.strip():
        analysis = ScamAnalysis.from_heuristics(message, memory)
        yield (
            render_analysis_html(message, analysis),
            render_memory_html(analysis, memory),
            memory,
            {},
            gr.update(interactive=True),
            gr.update(interactive=True),
            qr_status,
        )
        return

    if is_low_context_message(message):
        analysis = low_context_analysis(message, memory)
        entry = memory_entry_from_analysis(message, analysis)
        if not memory or memory[-1].get("text") != entry.get("text"):
            memory.append(entry)
        yield (
            render_analysis_html(message, analysis),
            render_memory_html(analysis, memory),
            memory,
            {"message": message, "entry": entry},
            gr.update(interactive=True),
            gr.update(interactive=True),
            qr_status,
        )
        return

    for active_step, progress in [(0, 12), (1, 30), (2, 54), (3, 76)]:
        yield (
            render_scanning_html(active_step=active_step, progress=progress),
            render_current_memory(memory),
            memory,
            {},
            gr.update(interactive=False),
            gr.update(interactive=False),
            qr_status,
        )
        sleep(0.25)

    try:
        analysis = run_analysis(message, memory)
    except Exception as exc:
        analysis = analysis_error(exc)
        yield (
            render_analysis_html(message, analysis),
            render_memory_html(analysis, memory),
            memory,
            {},
            gr.update(interactive=True),
            gr.update(interactive=True),
            qr_status,
        )
        return
    _save_confident_analysis(message, analysis)

    last_scan = {
        "message": message,
        "entry": memory_entry_from_analysis(message, analysis),
    }
    entry = last_scan["entry"]
    if not memory or memory[-1].get("text") != entry.get("text"):
        memory.append(entry)
    yield (
        render_analysis_html(message, analysis),
        render_memory_html(analysis, memory),
        memory,
        last_scan,
        gr.update(interactive=True),
        gr.update(interactive=True),
        qr_status,
    )


def analysis_error(exc: Exception) -> ScamAnalysis:
    return ScamAnalysis(
        risk_level="needs_check",
        scam_type="analysis_error",
        summary="PhishBrake could not finish the model scan in this environment.",
        tactics=["runtime error"],
        safest_action="Do not click links or reply yet. Verify through an official app, website, or known phone number.",
        trusted_person_message=f"Can you check this for me? PhishBrake hit an analysis error: {exc}",
        scam_dna={
            "Impersonates": "Unknown",
            "Pressure": "Unknown",
            "Ask": "Unknown",
            "Risk": "Could not analyze",
        },
    )


def remember_current(message: str, memory: list[dict] | None, last_scan: dict | None) -> tuple[str, str, list[dict]]:
    memory = memory or []
    if not message.strip():
        return render_save_status("Paste a message first."), render_current_memory(memory), memory

    last_scan = last_scan or {}
    entry = last_scan.get("entry")
    if last_scan.get("message") != message or not isinstance(entry, dict):
        return (
            render_save_status("Analyze this message first, then save the pattern."),
            render_current_memory(memory),
            memory,
        )

    if memory and memory[-1].get("text") == entry.get("text"):
        return (
            render_save_status("This pattern is already saved for this session."),
            render_current_memory(memory),
            memory,
        )

    memory.append(entry)
    return render_save_status("Saved this scam pattern for this session."), render_current_memory(memory), memory


def build_app() -> gr.Blocks:
    gr.set_static_paths(paths=[LOGO_PATH.resolve()])
    with gr.Blocks(title="PhishBrake") as demo:
        memory_state = gr.State([])
        last_scan_state = gr.State({})

        gr.HTML(
            f"""
            <header class="topbar">
              <div class="brand-lockup">
                <span class="brand-mark" aria-hidden="true">
                  <img src="/gradio_api/file=phishbrake_logo.png" alt="" />
                </span>
                <span class="brand-name">PHISHBRAKE</span>
              </div>
              <div class="status-tags">
                <span>[ SECURE_ENV: ACTIVE ]</span>
                <span>[ STATUS: READY ]</span>
                <span>[ BACKEND: {escape(backend_status_label())} ]</span>
                <span>[ MODEL: {escape(model_status_label())} ]</span>
                <span>[ INITIATIVE: ML_EMPOWERMENT_BUILD_CHALLENGE_3_0 ]</span>
              </div>
            </header>
            """
        )

        with gr.Row(elem_classes=["main-grid"]):
            with gr.Column(scale=5, elem_classes=["scan-panel"]):
                gr.HTML(
                    """
                    <div class="window-titlebar input-titlebar">
                      <span class="traffic-dots" aria-hidden="true"><i></i><i></i><i></i></span>
                      <span>Message to check</span>
                    </div>
                    <p class="panel-kicker">MESSAGE TO CHECK</p>
                    """
                )
                message = gr.Textbox(
                    label=None,
                    show_label=False,
                    placeholder="Paste a text, email, or DM here.",
                    lines=10,
                    max_lines=16,
                )
                qr_image = gr.Image(
                    type="pil",
                    label="Upload suspicious screenshot or QR code (optional)",
                    sources=["upload"],
                )
                qr_status = gr.Markdown("QR Code Decoded", visible=False)
                analyze = gr.Button("Check message", variant="primary", elem_classes=["check-btn"])
                gr.Examples(examples=EXAMPLES, inputs=message, label="Try a sample")
                gr.Examples(
                  examples=[[str(SAFE_QR_DEMO_PATH)], [str(PHISHING_QR_DEMO_PATH)]],
                  inputs=qr_image,
                  label="Safe and phishing QR samples",
                )
                memory = gr.HTML(render_current_memory([]))

            with gr.Column(scale=7):
                result = gr.HTML(
                    """
                    <div class="home-stack">
                      <section class="retro-window status-window">
                        <div class="window-titlebar"><span>Ready to check</span></div>
                        <div class="window-body status-body">
                          <p class="standing-by">SYSTEM STANDING BY</p>
                          <h2>PhishBrake is ready to shield your loved ones from digital fraud.</h2>
                          <p>Paste any text message, email, or DM on the left. The local model will evaluate risk factors, unpack the scam strategy, and deliver a plain-English protection plan.</p>
                        </div>
                      </section>
                      <section class="retro-window guide-window">
                        <div class="window-titlebar"><span>How to use it</span></div>
                        <div class="window-body guide-body">
                          <p>1. Copy a text message from your phone or an email that feels off.</p>
                          <p>2. Paste it into the input area on the left of this screen.</p>
                          <p>3. Click CHECK MESSAGE to analyze it with private local AI.</p>
                        </div>
                      </section>
                    </div>
                    """
                )
        analyze.click(
            fn=analyze_message,
            inputs=[message, memory_state, qr_image],
            outputs=[result, memory, memory_state, last_scan_state, message, analyze, qr_status],
            show_progress="hidden",
        )
        demo.load(
            fn=warm_model,
            inputs=None,
            outputs=None,
            show_progress="hidden",
            api_visibility="private",
        )
    return demo


if __name__ == "__main__":
    demo = build_app()
    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        theme=app_theme(),
        css=app_css(),
        favicon_path=str(LOGO_PATH),
        footer_links=[],
    )
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)
