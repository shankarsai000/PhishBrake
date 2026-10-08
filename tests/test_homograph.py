from jawbreaker.homograph import (
    audit_message_urls,
    detect_homoglyphs,
    detect_typosquatting,
    extract_urls,
)
from jawbreaker.analyzers import _build_chat_messages, repair_prediction


def test_detects_cyrillic_a_in_paypal_domain() -> None:
    domain = "pаypal.com"

    findings = detect_homoglyphs(domain)
    audit = audit_message_urls(f"Please review https://{domain}/login.")

    assert any(finding["type"] == "mixed_script" for finding in findings)
    assert audit["flagged_urls"][0]["domain"] == domain
    assert "System Guard" in audit["advisory"]


def test_typosquatting_flags_close_and_brand_prefixed_domains() -> None:
    assert detect_typosquatting("paypal.com") == []
    assert detect_typosquatting("paypa1.com")
    assert detect_typosquatting("paypa1.example")
    assert detect_typosquatting("wellsfarg0.com")
    assert detect_typosquatting("app1e.com")
    assert detect_typosquatting("usps-tracking-update.com")
    assert detect_typosquatting("usps-redelivery-notice.info")

    audit = audit_message_urls("Check https://paypal.com, not https://paypa1.com.")
    assert [finding["domain"] for finding in audit["flagged_urls"]] == ["paypa1.com"]
    assert audit["reasons"][0]["target"] == "paypal.com"


def test_punycode_domain_is_flagged() -> None:
    findings = detect_homoglyphs("xn--pple-43d.com")

    assert findings[0]["type"] == "punycode"


def test_system_guard_is_prepended_and_repair_rejects_safe_risk() -> None:
    message = "Log in at https://paypa1.com to verify your account."

    messages, homograph_detected = _build_chat_messages(message)
    prediction = repair_prediction(
        {
            "risk_level": "safe",
            "scam_type": "unknown",
            "summary": "No risk found.",
            "tactics": [],
            "safest_action": "Continue as usual.",
            "trusted_person_message": "This looks safe.",
            "scam_dna": {},
        },
        homograph_detected=homograph_detected,
    )

    assert homograph_detected
    assert messages[0]["content"].startswith(
        "[System Guard: Deceptive homograph/typosquat domain detected: paypa1.com]"
    )
    assert prediction["risk_level"] == "dangerous"
    assert prediction["scam_type"] == "homograph_phishing"
    assert prediction["summary"] == "This message contains a deceptive lookalike domain."
    assert "lookalike domain" in prediction["tactics"]
    assert "Do not click links" in prediction["safest_action"]


def test_plain_text_without_urls_has_no_findings() -> None:
    assert extract_urls("Please call the number you already trust.") == []
    assert audit_message_urls("Please call the number you already trust.") == {
        "urls": [],
        "flagged_urls": [],
        "reasons": [],
        "advisory": "",
    }
from jawbreaker.homograph import (
    audit_message_urls,
    detect_homoglyphs,
    detect_typosquatting,
    extract_urls,
)


def test_detects_cyrillic_a_in_paypal_domain() -> None:
    domain = "pаypal.com"

    findings = detect_homoglyphs(domain)
    audit = audit_message_urls(f"Please review https://{domain}/login.")

    assert any(finding["type"] == "mixed_script" for finding in findings)
    assert audit["flagged_urls"][0]["domain"] == domain
    assert "System Guard" in audit["advisory"]


def test_typosquatting_flags_close_and_brand_prefixed_domains() -> None:
    assert detect_typosquatting("paypal.com") == []
    assert detect_typosquatting("paypa1.com")
    assert detect_typosquatting("wellsfarg0.com")
    assert detect_typosquatting("app1e.com")
    assert detect_typosquatting("usps-tracking-update.com")

    audit = audit_message_urls("Check https://paypal.com, not https://paypa1.com.")
    assert [finding["domain"] for finding in audit["flagged_urls"]] == ["paypa1.com"]
    assert audit["reasons"][0]["target"] == "paypal.com"


def test_punycode_domain_is_flagged() -> None:
    findings = detect_homoglyphs("xn--pple-43d.com")

    assert findings[0]["type"] == "punycode"


def test_plain_text_without_urls_has_no_findings() -> None:
    assert extract_urls("Please call the number you already trust.") == []
    audit = audit_message_urls("Please call the number you already trust.")
    assert audit["urls"] == []
    assert audit["flagged_urls"] == []
    assert audit["reasons"] == []
    assert audit["advisory"] == ""
    assert audit["has_zero_width"] is False


def test_detects_zero_width_characters_and_strips_them() -> None:
    from jawbreaker.homograph import detect_zero_width_evasion, strip_zero_width_chars

    evasion_text = "P\u200ba\u200by\u200bP\u200ba\u200bl verification required"
    assert detect_zero_width_evasion(evasion_text) is True
    assert strip_zero_width_chars(evasion_text) == "PayPal verification required"

    audit = audit_message_urls(evasion_text)
    assert audit["has_zero_width"] is True
    assert any(reason["type"] == "zero_width_evasion" for reason in audit["reasons"])


def test_detects_userinfo_domain_spoofing_and_ip_hosts() -> None:
    from jawbreaker.homograph import detect_url_anomalies

    userinfo_url = "https://paypal.com@evil-phish.net/login"
    anomalies = detect_url_anomalies(userinfo_url)
    assert any(a["type"] == "userinfo_spoof" for a in anomalies)

    ip_url = "http://192.168.1.100/bank"
    ip_anomalies = detect_url_anomalies(ip_url)
    assert any(a["type"] == "ip_host" for a in ip_anomalies)

    shortener_url = "https://bit.ly/3xYz90"
    shortener_anomalies = detect_url_anomalies(shortener_url)
    assert any(a["type"] == "url_shortener" for a in shortener_anomalies)

