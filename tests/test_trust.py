from jawbreaker.schema import ScamAnalysis
from jawbreaker.trust import (
    build_trusted_note,
    confidence_metadata,
    is_low_context_message,
    low_context_analysis,
)


def test_low_context_message_surfaces_uncertainty() -> None:
    assert is_low_context_message("hi")
    assert not is_low_context_message("Pay now: http://example.test")

    analysis = low_context_analysis("hi")

    assert analysis.risk_level == "needs_check"
    assert analysis.scam_type == "missing_context"
    assert "Too little context" in analysis.summary
    assert analysis.scam_dna["Pressure"] == "Not enough context"


def test_confidence_metadata_uses_qualitative_language() -> None:
    analysis = ScamAnalysis(
        risk_level="dangerous",
        scam_type="family_impersonation",
        summary="This looks dangerous.",
        tactics=["impersonation", "secrecy", "payment pressure"],
        safest_action="Do not send money.",
        scam_dna={
            "Impersonates": "Family member",
            "Pressure": "Keep it secret",
            "Ask": "Send money",
            "Risk": "Payment theft",
        },
    )

    confidence = confidence_metadata(
        "Hi Grandma, I lost my phone. Can you send $800 today? Please don't tell Mom.",
        analysis,
    )

    assert confidence["label"] == "High confidence"
    assert "%" not in confidence["label"]
    assert "%" not in confidence["detail"]


def test_trusted_note_for_dangerous_message_is_actionable() -> None:
    message = "Hi Grandma, I lost my phone. Can you send $800 today? Please don't tell Mom."
    analysis = ScamAnalysis.from_heuristics(message)

    note = build_trusted_note(message, analysis)

    assert "Can you check this message with me before I do anything?" in note
    assert message in note
    assert "Safest next step:" in note
    assert "I have not clicked any links, replied, sent money, or shared codes." in note


def test_trusted_note_for_needs_check_asks_for_confirmation_outside_thread() -> None:
    analysis = ScamAnalysis(
        risk_level="needs_check",
        scam_type="possible_legitimate_alert",
        summary="Verify this through a trusted route.",
        tactics=["verification needed"],
        safest_action="Open the official app yourself.",
        scam_dna={
            "Impersonates": "Bank",
            "Pressure": "Routine alert",
            "Ask": "Verify",
            "Risk": "Unclear",
        },
    )

    note = build_trusted_note("Bank alert: new login. Open the official app.", analysis)

    assert "Can you help me verify this before I act?" in note
    assert "please confirm outside this message thread" in note
