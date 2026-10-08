from jawbreaker.schema import ScamAnalysis
from jawbreaker.analyzers import prediction_to_analysis
from jawbreaker.render import render_analysis_html, render_scanning_html


def test_family_impersonation_is_dangerous() -> None:
    analysis = ScamAnalysis.from_heuristics(
        "Hi Grandma, I lost my phone. This is my new number. Can you send $800 today?"
    )

    assert analysis.risk_level == "dangerous"
    assert analysis.scam_type == "family_impersonation"
    assert "Trust a new number" == analysis.scam_dna["Ask"]


def test_legitimate_fraud_alert_needs_check_not_dangerous() -> None:
    analysis = ScamAnalysis.from_heuristics(
        "Chase fraud alert: Did you attempt a $249.00 purchase at TARGET? Reply YES or NO."
    )

    assert analysis.risk_level == "needs_check"
    assert analysis.scam_type == "possible_legitimate_alert"


def test_device_protection_subscription_alert_is_tech_support() -> None:
    analysis = ScamAnalysis.from_heuristics(
        "Device protection team: subscription expired. Call the number in this alert to avoid data loss."
    )

    assert analysis.risk_level == "dangerous"
    assert analysis.scam_type == "tech_support"


def test_billing_link_with_account_closure_is_dangerous() -> None:
    analysis = ScamAnalysis.from_heuristics(
        "Streaming billing: payment failed and your account closes tonight. Update through this message link: http://billing-review.example"
    )

    assert analysis.risk_level == "dangerous"
    assert analysis.scam_type == "credential_theft"


def test_prediction_to_analysis_normalizes_model_json() -> None:
    analysis = prediction_to_analysis(
        {
            "risk_level": "not_valid",
            "scam_type": "package_phishing",
            "summary": "Fake delivery notice.",
            "tactics": ["fake authority"],
            "safest_action": "Open the official carrier website yourself.",
            "trusted_person_message": "Can you check this?",
            "scam_dna": {
                "impersonates": "USPS",
                "pressure": "Held package",
                "ask": "Open link",
                "risk": "credential theft",
            },
        },
        similar_memory="This resembles a saved pattern.",
    )

    assert analysis.risk_level == "needs_check"
    assert analysis.scam_dna["Impersonates"] == "USPS"
    assert analysis.similar_memory == "This resembles a saved pattern."


def test_render_humanizes_model_style_labels() -> None:
    analysis = ScamAnalysis(
        risk_level="dangerous",
        scam_type="credential_theft",
        summary="This message mentions credential_theft.",
        tactics=["unknown_urgent_action", "unknown_urgent_payment", "credential_theft"],
        safest_action="Do not click the link.",
        scam_dna={
            "Impersonates": "legitimate_company",
            "Pressure": "sense_of_urgency",
            "Ask": "click_link_and_verify",
            "Risk": "credential_theft",
        },
    )

    html = render_analysis_html("USPS: verify now", analysis)

    assert "legitimate company" in html
    assert "click a link and verify" in html
    assert "unknown_urgent_action" not in html
    assert "urgent payment" in html
    assert "data-copy=" in html


def test_render_scanning_html_tracks_progress() -> None:
    html = render_scanning_html(active_step=2, progress=54)

    assert "54% COMPLETE" in html
    assert "✓ Preparing the safety model" in html
    assert "> Checking scam pressure signals" in html
def test_safe_verification_code_notice_is_not_treated_as_credential_theft() -> None:
    from jawbreaker.schema import ScamAnalysis

    analysis = ScamAnalysis.from_heuristics(
        "Your verification code is 482913. Do not share this code with anyone. "
        "We will never ask for it by phone."
    )

    assert analysis.risk_level == "safe"
    assert analysis.scam_type == "none"
