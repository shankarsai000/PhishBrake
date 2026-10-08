from dataclasses import asdict

import app
from jawbreaker import cache
from jawbreaker.schema import ScamAnalysis


def test_save_and_lookup_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "models" / "threat_cache.db")
    text = "Your bank account is locked. Verify through this suspicious link."
    analysis = {
        "risk_level": "dangerous",
        "scam_type": "credential_phishing",
        "summary": "The message asks for account verification through a suspicious link.",
    }

    assert cache.lookup_cache(text) is None
    cache.save_cache(text, analysis)

    assert cache.CACHE_DB_PATH.exists()
    assert cache.lookup_cache(text) == analysis


def test_normalization_removes_variable_otp_phone_and_amount_values() -> None:
    first = "Bank alert: OTP 123456. Send $250 to +1 (415) 555-0199 now."
    second = "BANK ALERT: OTP 987654. Send $75 to +1 (212) 555-0113 now."

    assert cache.normalize_text(first) == cache.normalize_text(second)


def test_normalized_variant_retrieves_same_cache_entry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "threat_cache.db")
    first = "Bank alert: OTP 123456. Send $250 to +1 (415) 555-0199 now."
    variant = "BANK ALERT: OTP 987654. Send $75 to +1 (212) 555-0113 now."
    analysis = {"risk_level": "dangerous", "summary": "Deceptive bank alert."}

    cache.save_cache(first, analysis)

    assert cache.lookup_cache(variant) == analysis


def test_analysis_payload_returns_cache_hit_without_running_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "threat_cache.db")
    text = "Bank alert: verify your account through this link immediately."
    analysis = ScamAnalysis(
        risk_level="dangerous",
        scam_type="credential_phishing",
        summary="The alert uses a suspicious account-verification link.",
        safest_action="Do not click links.",
    )
    cache.save_cache(text, asdict(analysis))
    monkeypatch.setattr("app.run_analysis", lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")))

    payload = app.analysis_payload(text, [])

    assert payload["analysis"]["risk_level"] == "dangerous"
    assert payload["cache_status"] == "⚡ Instant Edge Cache Hit"


def test_native_gradio_callback_returns_cached_result_before_progress(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "threat_cache.db")
    text = "Bank alert: verify your account through this link immediately."
    analysis = ScamAnalysis(
        risk_level="dangerous",
        scam_type="credential_phishing",
        summary="The alert uses a suspicious account-verification link.",
        safest_action="Do not click links.",
    )
    cache.save_cache(text, asdict(analysis))
    monkeypatch.setattr("app.run_analysis", lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")))

    outputs = next(app.analyze_message(text, [], None))

    assert "⚡ Instant Edge Cache Hit" in outputs[0]
    assert outputs[2][-1]["text"] == text


def test_analysis_payload_saves_confident_but_not_needs_check_results(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DB_PATH", tmp_path / "threat_cache.db")
    confident_text = "A clearly suspicious payment request that needs a model result."
    confident = ScamAnalysis(
        risk_level="suspicious",
        scam_type="payment_request",
        summary="The request should be verified.",
        safest_action="Verify through the official app.",
    )
    monkeypatch.setattr("app.run_analysis", lambda *_: confident)

    app.analysis_payload(confident_text, [])
    assert cache.lookup_cache(confident_text) == asdict(confident)

    uncertain_text = "A separate uncertain message that needs a model result."
    uncertain = ScamAnalysis(
        risk_level="needs_check",
        scam_type="unknown",
        summary="More context is needed.",
        safest_action="Verify through an official channel.",
    )
    monkeypatch.setattr("app.run_analysis", lambda *_: uncertain)

    app.analysis_payload(uncertain_text, [])
    assert cache.lookup_cache(uncertain_text) is None
