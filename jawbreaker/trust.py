from __future__ import annotations

import re

from jawbreaker.schema import ScamAnalysis


LOW_CONTEXT_SIGNALS = [
    "http://",
    "https://",
    "$",
    "pay",
    "payment",
    "send",
    "wire",
    "gift card",
    "crypto",
    "code",
    "pin",
    "password",
    "verify",
    "urgent",
    "immediately",
    "call",
    "reply",
    "new number",
    "don't tell",
    "dont tell",
    "account",
    "bank",
    "card",
]


def is_low_context_message(message: str) -> bool:
    cleaned = " ".join(message.strip().split())
    if not cleaned:
        return False

    text = cleaned.lower()
    if any(signal in text for signal in LOW_CONTEXT_SIGNALS):
        return False

    words = re.findall(r"[a-z0-9']+", text)
    return len(cleaned) < 24 or len(words) < 4


def low_context_analysis(message: str, memory: list[dict] | None = None) -> ScamAnalysis:
    memory = memory or []
    text = message.lower()
    return ScamAnalysis(
        risk_level="needs_check",
        scam_type="missing_context",
        summary="Too little context to judge this message safely.",
        tactics=["missing context"],
        safest_action=(
            "Ask who sent it or verify through a known contact before clicking, replying, sending money, "
            "or sharing codes."
        ),
        trusted_person_message=(
            "Can you help me check this before I act?\n\n"
            f"Message I received:\n\"{' '.join(message.strip().split())}\"\n\n"
            "PhishBrake says there is not enough context to judge it safely. If this came from you, "
            "please confirm outside this message thread."
        ),
        scam_dna={
            "Impersonates": "Unknown sender",
            "Pressure": "Not enough context",
            "Ask": "No clear ask shown",
            "Risk": "Could not judge safely",
        },
        similar_memory=_find_similar_memory(text, memory),
    )


def confidence_metadata(message: str, analysis: ScamAnalysis) -> dict[str, str]:
    if is_low_context_message(message) or analysis.scam_type == "missing_context":
        return {
            "label": "Needs more context",
            "detail": "The message is too short to judge by itself.",
        }

    if analysis.risk_level == "dangerous":
        evidence_count = len(analysis.tactics) + _filled_dna_count(analysis)
        if evidence_count >= 5:
            return {
                "label": "High confidence",
                "detail": "Multiple warning signs line up.",
            }
        return {
            "label": "Strong warning",
            "detail": "Treat it as unsafe until verified another way.",
        }

    if analysis.risk_level == "suspicious":
        return {
            "label": "Some warning signs",
            "detail": "There is enough risk to pause and verify.",
        }

    if analysis.risk_level == "needs_check":
        return {
            "label": "Needs more context",
            "detail": "Could be real, but verify through a trusted route.",
        }

    return {
        "label": "No strong scam signs",
        "detail": "Still avoid unexpected links, codes, or payment asks.",
    }


def build_trusted_note(message: str, analysis: ScamAnalysis) -> str:
    cleaned = " ".join(message.strip().split())
    if len(cleaned) > 500:
        cleaned = cleaned[:497].rstrip() + "..."

    risk = analysis.risk_level.replace("_", " ")
    scam_type = analysis.scam_type.replace("_", " ")

    if analysis.risk_level == "safe":
        return (
            "Can you sanity-check this with me?\n\n"
            f"Message I received:\n\"{cleaned}\"\n\n"
            "PhishBrake did not find strong scam signs, but I want to be careful before I act."
        )

    if analysis.risk_level in {"needs_check", "suspicious"}:
        return (
            "Can you help me verify this before I act?\n\n"
            f"Message I received:\n\"{cleaned}\"\n\n"
            f"PhishBrake marked it as {risk}"
            f"{f' ({scam_type})' if scam_type and scam_type != 'none' else ''}.\n"
            f"Safest next step: {analysis.safest_action}\n\n"
            "If you know whether this is real, please confirm outside this message thread."
        )

    return (
        "Can you check this message with me before I do anything?\n\n"
        f"Message I received:\n\"{cleaned}\"\n\n"
        f"PhishBrake marked it as dangerous"
        f"{f' ({scam_type})' if scam_type and scam_type != 'none' else ''}.\n"
        f"Safest next step: {analysis.safest_action}\n\n"
        "I have not clicked any links, replied, sent money, or shared codes."
    )


def _filled_dna_count(analysis: ScamAnalysis) -> int:
    empty_values = {"", "unknown", "unknown sender", "not obvious", "no direct ask found", "none"}
    return sum(1 for value in analysis.scam_dna.values() if value.strip().lower() not in empty_values)


def _find_similar_memory(text: str, memory: list[dict]) -> str:
    tokens = set(text.split())
    best_score = 0.0
    best = ""
    for item in memory:
        old_tokens = set(str(item.get("text", "")).lower().split())
        if not old_tokens:
            continue
        score = len(tokens & old_tokens) / max(len(tokens | old_tokens), 1)
        if score > best_score:
            best_score = score
            best = str(item.get("summary", "a previous saved pattern"))
    if best_score >= 0.18:
        return f"This resembles a saved pattern: {best}"
    return ""
