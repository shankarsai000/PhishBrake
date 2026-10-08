from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from typing import Any

from jawbreaker.contract import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from jawbreaker.homograph import audit_message_urls
from jawbreaker.schema import ScamAnalysis


Prediction = dict[str, Any]
Analyzer = Callable[[str], Prediction]
DEFAULT_PHISHING_CLASSIFIER_ID = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
PHISHING_CLASS_LABELS = (
    "legitimate_email",
    "phishing_url",
    "legitimate_url",
    "phishing_url_alt",
)


def phishing_score_from_probabilities(probabilities: list[float]) -> float:
    if len(probabilities) != len(PHISHING_CLASS_LABELS):
        raise ValueError("phishing classifier must return four class probabilities")
    return probabilities[1] + probabilities[3]


def _build_chat_messages(
    message: str,
    system_context: str | None = None,
) -> tuple[list[dict[str, str]], bool]:
    audit = audit_message_urls(message)
    system_context_parts = []
    if audit["advisory"]:
        system_context_parts.append(audit["advisory"])
    if system_context:
        system_context_parts.append(system_context)
    system_context_parts.append(SYSTEM_PROMPT)
    system_prompt = "\n\n".join(system_context_parts)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(message=message)},
    ]
    return messages, bool(audit["flagged_urls"])


UNSAFE_ACTION_PHRASES = [
    "click the link",
    "open the link",
    "use the link",
    "call the number in the message",
    "call the number shown",
    "call the number in this alert",
    "reply with your code",
    "send the code",
    "enter your password",
    "enter your online banking password",
    "enter your credentials",
    "submit your password",
    "provide your password",
    "send money",
    "wire money",
    "wire payment",
    "send funds",
    "send crypto",
    "buy gift cards",
    "send gift cards",
    "text me the codes",
    "share your password",
    "share your pin",
    "share the login code",
    "install remote access",
]

SAFE_ACTION_BY_RISK = {
    "dangerous": (
        "Do not click links, do not reply, and do not send money. Contact the company or person using an "
        "official app, official website, or a number you already trust."
    ),
    "suspicious": (
        "Pause before acting. Do not use links or phone numbers from the message. Verify through an official "
        "app, official website, or known phone number."
    ),
    "needs_check": "Verify directly through the official app, official website, or a known phone number.",
    "safe": "If this came from someone you know and asks for nothing sensitive, it is probably safe.",
}


def has_unsafe_action(action: str) -> bool:
    text = f" {action.lower()} "
    safe_prefixes = ["do not", "don't", "never", "avoid", "do n't"]
    for phrase in UNSAFE_ACTION_PHRASES:
        if phrase not in text:
            continue
        if any(f"{prefix} {phrase}" in text for prefix in safe_prefixes):
            continue
        return True
    return False


def analysis_to_prediction(analysis: ScamAnalysis) -> Prediction:
    return {
        "risk_level": analysis.risk_level,
        "scam_type": analysis.scam_type,
        "summary": analysis.summary,
        "tactics": analysis.tactics,
        "safest_action": analysis.safest_action,
        "trusted_person_message": analysis.trusted_person_message,
        "scam_dna": {
            "impersonates": analysis.scam_dna.get("Impersonates", ""),
            "pressure": analysis.scam_dna.get("Pressure", ""),
            "ask": analysis.scam_dna.get("Ask", ""),
            "risk": analysis.scam_dna.get("Risk", ""),
        },
    }


def prediction_to_analysis(prediction: Prediction, *, similar_memory: str = "") -> ScamAnalysis:
    risk_level = prediction.get("risk_level")
    if risk_level not in {"dangerous", "suspicious", "needs_check", "safe"}:
        risk_level = "needs_check"

    tactics = prediction.get("tactics", [])
    if not isinstance(tactics, list):
        tactics = []

    scam_dna = prediction.get("scam_dna", {})
    if not isinstance(scam_dna, dict):
        scam_dna = {}

    return ScamAnalysis(
        risk_level=str(risk_level),
        scam_type=str(prediction.get("scam_type", "unknown")),
        summary=str(prediction.get("summary", "This message should be checked before anyone acts.")),
        tactics=[str(tactic) for tactic in tactics],
        safest_action=str(
            prediction.get(
                "safest_action",
                "Do not click links or reply. Verify through an official app, website, or known phone number.",
            )
        ),
        trusted_person_message=str(
            prediction.get(
                "trusted_person_message",
                "Can you check this for me before I respond or click anything?",
            )
        ),
        scam_dna={
            "Impersonates": str(scam_dna.get("impersonates", "")),
            "Pressure": str(scam_dna.get("pressure", "")),
            "Ask": str(scam_dna.get("ask", "")),
            "Risk": str(scam_dna.get("risk", "")),
        },
        similar_memory=similar_memory,
    )


def repair_prediction(prediction: Prediction, *, homograph_detected: bool = False) -> Prediction:
    repaired = dict(prediction)
    risk_level = repaired.get("risk_level")
    if homograph_detected:
        risk_level = "dangerous"
        repaired["risk_level"] = risk_level
        repaired["scam_type"] = "homograph_phishing"
        repaired["summary"] = "This message contains a deceptive lookalike domain."
        tactics = repaired.get("tactics")
        repaired["tactics"] = list(dict.fromkeys([*tactics, "lookalike domain"]))
        repaired["trusted_person_message"] = (
            "Can you help me verify this through the official app or a known website before I interact with it?"
        )
        scam_dna = repaired.get("scam_dna")
        scam_dna = scam_dna if isinstance(scam_dna, dict) else {}
        repaired["scam_dna"] = {
            **scam_dna,
            "impersonates": scam_dna.get("impersonates") or "a trusted website",
            "pressure": scam_dna.get("pressure") or "deceptive lookalike domain",
            "ask": scam_dna.get("ask") or "visit the suspicious domain",
            "risk": scam_dna.get("risk") or "credential theft or account compromise",
        }
    elif risk_level not in {"dangerous", "suspicious", "needs_check", "safe"}:
        risk_level = "needs_check"
        repaired["risk_level"] = risk_level

    if not isinstance(repaired.get("scam_type"), str):
        repaired["scam_type"] = "unknown"
    if not isinstance(repaired.get("tactics"), list):
        repaired["tactics"] = []
    if not isinstance(repaired.get("scam_dna"), dict):
        repaired["scam_dna"] = {}

    if not isinstance(repaired.get("safest_action"), str):
        repaired["safest_action"] = (
            "Do not click links or reply. Verify through an official app, website, or known phone number."
        )
    if homograph_detected:
        repaired["safest_action"] = SAFE_ACTION_BY_RISK["dangerous"]
    elif has_unsafe_action(str(repaired.get("safest_action", ""))):
        repaired["safest_action"] = SAFE_ACTION_BY_RISK[str(risk_level)]
    if not isinstance(repaired.get("trusted_person_message"), str):
        repaired["trusted_person_message"] = "Can you check this for me before I respond or click anything?"
    if not isinstance(repaired.get("summary"), str):
        scam_type = str(repaired.get("scam_type") or "unknown").replace("_", " ")
        if risk_level == "dangerous":
            repaired["summary"] = f"This looks dangerous: likely {scam_type}."
        elif risk_level == "suspicious":
            repaired["summary"] = f"This has warning signs of {scam_type}."
        elif risk_level == "needs_check":
            repaired["summary"] = "This should be verified through a trusted route before acting."
        else:
            repaired["summary"] = "No strong scam pattern was found."
    return repaired


def heuristic_analyzer(message: str) -> Prediction:
    return analysis_to_prediction(ScamAnalysis.from_heuristics(message))


def has_high_confidence_danger_signal(message: str, heuristic: ScamAnalysis) -> bool:
    text = message.lower()
    scam_type = heuristic.scam_type
    tactics = set(heuristic.tactics)

    high_confidence_types = {
        "callback_phishing",
        "payment_request",
        "prize_scam",
        "tech_support",
    }
    if scam_type in high_confidence_types:
        return True

    if scam_type == "family_impersonation":
        has_identity_shift = any(
            token in text
            for token in [
                "new number",
                "changed numbers",
                "number only",
                "phone broke",
                "temporary phone",
                "lost my phone",
            ]
        )
        has_secrecy = any(token in text for token in ["don't tell", "dont tell", "do not tell", "keep this quiet"])
        has_sensitive_ask = any(
            token in text
            for token in [
                "wire",
                "gift card",
                "send money",
                "send funds",
                "can you send",
                "rent",
                "hospital bill",
                "bail",
                "emergency",
            ]
        )
        return has_identity_shift or has_secrecy or has_sensitive_ask

    if scam_type == "job_scam":
        return any(
            token in text
            for token in [
                "telegram",
                "whatsapp",
                "training starts",
                "pay is instant",
                "per day",
                "payment is made immediately",
                "deposit",
                "onboarding fee",
                "training fee",
                "equipment shipment",
                "payroll setup",
            ]
        )

    if scam_type == "investment_scam":
        return any(
            token in text
            for token in [
                "wrong number",
                "wrong chat",
                "wrong contact",
                "wrong recipient",
                "meant to reach",
                "thought this was",
            ]
        ) and any(
            token in text
            for token in [
                "investment",
                "trading",
                "crypto",
                "wallet",
                "exchange",
                "gold futures",
                "gold strategy",
                "currency signal",
                "coin signal",
                "guaranteed returns",
                "low-risk",
                "one spot just opened",
                "reserve you a spot",
                "window closes",
                "small deposit",
                "app link",
                "open the app",
            ]
        )

    if scam_type == "credential_theft":
        if any(
            token in text
            for token in [
                "do not share this code",
                "do not share your code",
                "we will never ask for it",
                "open the official app",
                "open the app directly",
            ]
        ):
            return False
        has_link_or_callback = any(token in text for token in ["http://", "https://", "call the number", "call this"])
        has_sensitive_secret = any(
            token in text
            for token in [
                "password",
                "pin",
                "one-time code",
                "verification code",
                "card number",
                "bank login",
                "bank details",
                "seed phrase",
                "upload id",
                "insurance number",
                "payment card",
                "account reset code",
                "reset code",
            ]
        )
        asks_for_secret = any(
            token in text
            for token in [
                "send your",
                "send the",
                "reply with",
                "enter your",
                "enter the",
                "provide your",
                "provide the",
                "share your",
                "share the",
                "read your",
                "read the",
                "verify your",
                "confirm your",
            ]
        )
        return has_link_or_callback or (has_sensitive_secret and asks_for_secret)

    return "payment pressure" in tactics and any(
        token in text
        for token in [
            "gift card",
            "wire",
            "crypto",
            "send money",
            "send funds",
            "deposit a check",
            "refund the difference",
            "courier fee",
            "service fee",
        ]
    )


def has_moderate_confidence_warning_signal(message: str, heuristic: ScamAnalysis) -> bool:
    text = message.lower()
    if heuristic.risk_level == "suspicious":
        return any(
            token in text
            for token in [
                "wrong number",
                "wrong chat",
                "wrong person",
                "wrong contact",
                "wrong recipient",
                "http://",
                "https://",
                "call this",
                "call the number",
                "reply yes",
                "reply no",
                "held package",
                "unpaid",
                "verify now",
                "confirm your",
                "account",
                "payment",
                "fee",
            ]
        )
    return heuristic.risk_level == "needs_check"


def has_high_confidence_benign_signal(message: str, heuristic: ScamAnalysis) -> bool:
    if heuristic.risk_level != "safe":
        return False

    text = message.lower()
    safe_code_notice = (
        "security code" in text
        and "if you did not request this code" in text
        and "no action is required" in text
        and not any(
            token in text
            for token in [
                "http://",
                "https://",
                "share this code",
                "send this code",
                "reply with",
                "enter the code",
                "read the code",
            ]
        )
    )
    if safe_code_notice:
        return True

    official_route_notice = (
        any(token in text for token in ["patient portal", "official app", "official website", "usual patient portal"])
        and any(token in text for token in ["sign in through", "use your usual", "normal patient portal", "open the official app"])
        and not any(token in text for token in ["reply with", "enter the code", "share this code", "send this code", "verify your", "confirm your"])
    )
    if official_route_notice:
        return False

    risky_tokens = [
        "http://",
        "https://",
        "reply",
        "call this",
        "call the number",
        "verification code",
        "one-time code",
        "password",
        "pin",
        "card number",
        "bank login",
        "send money",
        "send funds",
        "wire",
        "gift card",
        "crypto",
        "wallet",
        "investment",
        "trading",
        "pay at",
        "pay through",
        "payment link",
        "enter card",
        "card details",
        "new number",
        "do not tell",
        "don't tell",
        "keep this quiet",
    ]
    if any(token in text for token in risky_tokens):
        return False

    routine_family = any(token in text for token in ["bringing soup", "dinner", "no need to cook", "family dinner"])
    routine_school = any(
        token in text
        for token in [
            "school reminder",
            "office reminder",
            "early pickup",
            "usual id",
            "front office",
            "lunch account notices go home",
            "backpacks today",
            "after-school club",
        ]
    )
    routine_health = any(
        token in text
        for token in [
            "prescription is ready",
            "prescription pickup",
            "call the store number on your bottle",
            "usual patient portal",
            "appointment is confirmed",
        ]
    )
    routine_public = any(token in text for token in ["library book", "weather alert", "trash pickup"])

    return routine_family or routine_school or routine_health or routine_public


def should_apply_heuristic_guard(
    message: str,
    model_analysis: ScamAnalysis,
    heuristic: ScamAnalysis,
    validation_errors: Iterable[str],
) -> bool:
    if list(validation_errors):
        return True

    risk_rank = {"safe": 0, "needs_check": 1, "suspicious": 2, "dangerous": 3}
    model_rank = risk_rank[model_analysis.risk_level]
    heuristic_rank = risk_rank[heuristic.risk_level]

    if model_rank < heuristic_rank:
        if heuristic.risk_level == "safe":
            return False
        if heuristic.risk_level == "dangerous":
            return has_high_confidence_danger_signal(message, heuristic)
        return has_moderate_confidence_warning_signal(message, heuristic)

    if model_rank > risk_rank["safe"] and heuristic.risk_level == "safe":
        return has_high_confidence_benign_signal(message, heuristic)

    if heuristic.risk_level == "safe":
        return False

    dna_values = [value.strip() for value in model_analysis.scam_dna.values()]
    has_dna = any(dna_values)
    has_tactics = bool(model_analysis.tactics)
    if has_dna or has_tactics:
        return False

    if heuristic.risk_level == "dangerous":
        return has_high_confidence_danger_signal(message, heuristic)
    return True


def load_prediction_jsonl(path: Path) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row.get("id")
        prediction = row.get("prediction")
        if not isinstance(case_id, str) or not isinstance(prediction, dict):
            raise ValueError(f"{path}:{line_number}: expected fields 'id' and object 'prediction'")
        predictions[case_id] = prediction
    return predictions


def prediction_file_analyzer(predictions: dict[str, Prediction], case_id: str) -> Prediction:
    if case_id not in predictions:
        raise KeyError(f"Missing prediction for case id: {case_id}")
    return predictions[case_id]


def load_json_prediction(content: str) -> Prediction:
    content = strip_thinking_tokens(content)
    try:
        prediction = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            prediction = recover_prediction_fields(content)
        else:
            try:
                prediction = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                prediction = recover_prediction_fields(content[start : end + 1])
    if not isinstance(prediction, dict):
        raise ValueError("model response must be a JSON object")
    return prediction


def recover_prediction_fields(content: str) -> Prediction:
    """Recover a usable prediction from near-JSON small-model output."""
    prediction: Prediction = {}

    string_fields = [
        "risk_level",
        "scam_type",
        "safest_action",
        "trusted_person_message",
        "summary",
    ]
    for field in string_fields:
        value = extract_json_string_field(content, field)
        if value:
            prediction[field] = value

    tactics_match = re.search(r'"tactics"\s*:\s*\[(.*?)\]', content, flags=re.DOTALL)
    if tactics_match:
        prediction["tactics"] = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', tactics_match.group(1))

    scam_dna: dict[str, str] = {}
    for key in ["impersonates", "pressure", "ask", "risk"]:
        value = extract_json_string_field(content, key)
        if value:
            scam_dna[key] = value
    if scam_dna:
        prediction["scam_dna"] = scam_dna

    if not prediction:
        raise json.JSONDecodeError("could not recover prediction fields", content, 0)
    return prediction


def extract_json_string_field(content: str, field: str) -> str | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"', content, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1).replace('\\"', '"').replace("\\n", " ").strip()


def strip_thinking_tokens(content: str) -> str:
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    return content.replace("<think>", "").strip()


def build_llama_cpp_analyzer(
    model_path: Path,
    *,
    chat_format: str | None = None,
    n_ctx: int = 4096,
    n_threads: int | None = None,
    n_gpu_layers: int = 0,
    n_batch: int = 512,
    n_ubatch: int = 512,
    offload_kqv: bool = True,
    op_offload: bool | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    system_context_provider: Callable[[str], str] | None = None,
) -> Analyzer:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            "llama-cpp-python is not installed. Install it before running --backend llama-cpp."
        ) from exc

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": n_batch,
        "n_ubatch": n_ubatch,
        "offload_kqv": offload_kqv,
        "verbose": False,
    }
    if op_offload is not None:
        kwargs["op_offload"] = op_offload
    if chat_format:
        kwargs["chat_format"] = chat_format
    if n_threads is not None:
        kwargs["n_threads"] = n_threads

    llm = Llama(**kwargs)

    def analyze(message: str) -> Prediction:
        system_context = system_context_provider(message) if system_context_provider else None
        messages, homograph_detected = _build_chat_messages(message, system_context)
        response = llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response["choices"][0]["message"]["content"]
        return repair_prediction(
            load_json_prediction(content),
            homograph_detected=homograph_detected,
        )

    return analyze


def _load_local_phishing_classifier(model_id: str) -> Callable[[str], float]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required for JAWBREAKER_BACKEND=hybrid."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to("cpu")
    model.eval()
    if getattr(model.config, "num_labels", None) != len(PHISHING_CLASS_LABELS):
        raise ValueError(
            f"{model_id} must expose {len(PHISHING_CLASS_LABELS)} classification labels."
        )

    def phishing_score(message: str) -> float:
        inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).tolist()
        return phishing_score_from_probabilities([float(value) for value in probabilities])

    return phishing_score


def _apply_phishing_guard(
    prediction: Prediction,
    phishing_score: float,
    *,
    phishing_threshold: float,
    suspicious_threshold: float,
) -> Prediction:
    repaired = repair_prediction(prediction)
    if phishing_score < suspicious_threshold:
        return repaired

    risk_level = "dangerous" if phishing_score >= phishing_threshold else "suspicious"
    risk_rank = {"safe": 0, "needs_check": 1, "suspicious": 2, "dangerous": 3}
    if risk_rank[repaired["risk_level"]] > risk_rank[risk_level]:
        return repaired

    repaired["risk_level"] = risk_level
    repaired["scam_type"] = "phishing"
    repaired["summary"] = (
        "A local phishing classifier found strong phishing indicators."
        if risk_level == "dangerous"
        else "A local phishing classifier found warning signs; verify before acting."
    )
    repaired["tactics"] = list(dict.fromkeys([*repaired["tactics"], "phishing classifier signal"]))
    repaired["safest_action"] = SAFE_ACTION_BY_RISK[risk_level]
    repaired["trusted_person_message"] = (
        "Can you help me verify this through an official app or a known website before I interact with it?"
    )
    scam_dna = repaired["scam_dna"]
    scam_dna.update(
        {
            "impersonates": scam_dna.get("impersonates") or "a trusted organization",
            "pressure": scam_dna.get("pressure") or "phishing indicators detected by the local classifier",
            "ask": scam_dna.get("ask") or "interact with the message or its URL",
            "risk": scam_dna.get("risk") or "credential theft or account compromise",
        }
    )
    return repaired


def build_hybrid_analyzer(
    model_path: Path,
    *,
    classifier_model_id: str = DEFAULT_PHISHING_CLASSIFIER_ID,
    phishing_threshold: float = 0.75,
    suspicious_threshold: float = 0.50,
    classifier: Callable[[str], float] | None = None,
    **llama_kwargs: Any,
) -> Analyzer:
    """Combine a local phishing classifier guard with the local GGUF explainer."""
    if not 0 <= suspicious_threshold <= phishing_threshold <= 1:
        raise ValueError("classifier thresholds must satisfy 0 <= suspicious <= phishing <= 1")

    score_context: ContextVar[float] = ContextVar("jawbreaker_phishing_score", default=0.0)
    classify = classifier or _load_local_phishing_classifier(classifier_model_id)

    def system_context_provider(_: str) -> str:
        score = score_context.get()
        return (
            "A local phishing classifier evaluated the original message. Its combined phishing score is "
            f"{score:.3f}. Treat this as supporting evidence, not ground truth. Explain observable warning signs "
            "in plain language and never recommend following suspicious links or sharing credentials."
        )

    explain = build_llama_cpp_analyzer(
        model_path,
        system_context_provider=system_context_provider,
        **llama_kwargs,
    )

    def analyze(message: str) -> Prediction:
        try:
            phishing_score = classify(message)
        except Exception as exc:
            print(f"jawbreaker classifier_error fallback=explainer error={exc!r}", flush=True)
            phishing_score = 0.0
        token = score_context.set(phishing_score)
        try:
            prediction = explain(message)
        finally:
            score_context.reset(token)
        return _apply_phishing_guard(
            prediction,
            phishing_score,
            phishing_threshold=phishing_threshold,
            suspicious_threshold=suspicious_threshold,
        )

    return analyze


def build_transformers_analyzer(
    model_id: str,
    *,
    adapter_id: str | None = None,
    max_new_tokens: int = 192,
    temperature: float = 0.0,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
) -> Analyzer:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "torch and transformers are required for --backend transformers / ZeroGPU."
        ) from exc

    started = perf_counter()
    print(
        "jawbreaker transformers load_start "
        f"model_id={model_id} adapter_id={adapter_id} device_map={device_map} dtype={dtype} "
        f"trust_remote_code={trust_remote_code} attn_implementation={attn_implementation}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )
    if adapter_id:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("peft is required when using a Transformers LoRA adapter.") from exc
        print(f"jawbreaker transformers adapter_load_start adapter_id={adapter_id}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_id)
        print(f"jawbreaker transformers adapter_load_ready adapter_id={adapter_id}", flush=True)
    model.eval()
    if temperature <= 0 and hasattr(model, "generation_config"):
        # MiniCPM5 ships sampling defaults. Jawbreaker uses deterministic decoding,
        # so clear them to avoid noisy Transformers warnings in Space logs.
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    device = getattr(model, "device", "unknown")
    hf_device_map = getattr(model, "hf_device_map", None)
    print(
        "jawbreaker transformers load_ready "
        f"elapsed={perf_counter() - started:.2f}s device={device} hf_device_map={hf_device_map}",
        flush=True,
    )

    def analyze(message: str) -> Prediction:
        messages, homograph_detected = _build_chat_messages(message)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        inputs.pop("token_type_ids", None)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        content = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return repair_prediction(
            load_json_prediction(content),
            homograph_detected=homograph_detected,
        )

    return analyze


def validate_prediction(prediction: Prediction) -> list[str]:
    errors = []
    required = {
        "risk_level",
        "scam_type",
        "summary",
        "tactics",
        "safest_action",
        "trusted_person_message",
        "scam_dna",
    }
    missing = required - set(prediction)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if prediction.get("risk_level") not in {"dangerous", "suspicious", "needs_check", "safe"}:
        errors.append(f"invalid risk_level: {prediction.get('risk_level')}")
    if not isinstance(prediction.get("tactics"), list):
        errors.append("tactics must be a list")
    if not isinstance(prediction.get("scam_dna"), dict):
        errors.append("scam_dna must be an object")
    for key in ["scam_type", "summary", "safest_action", "trusted_person_message"]:
        if key in prediction and not isinstance(prediction[key], str):
            errors.append(f"{key} must be a string")
    return errors


def write_predictions(path: Path, rows: Iterable[dict], predictions: dict[str, Prediction]) -> None:
    lines = []
    for row in rows:
        case_id = row["id"]
        lines.append(json.dumps({"id": case_id, "prediction": predictions[case_id]}, ensure_ascii=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
