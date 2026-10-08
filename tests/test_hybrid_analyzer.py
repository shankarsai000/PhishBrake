from pathlib import Path

import app
from jawbreaker import analyzers


SAFE_PREDICTION = {
    "risk_level": "safe",
    "scam_type": "none",
    "summary": "No strong scam pattern was found.",
    "tactics": [],
    "safest_action": "Verify through the official app if unsure.",
    "trusted_person_message": "Can you double-check this with me?",
    "scam_dna": {"impersonates": "", "pressure": "", "ask": "", "risk": ""},
}


def test_phishing_score_sums_both_phishing_classes() -> None:
    assert analyzers.phishing_score_from_probabilities([0.1, 0.4, 0.2, 0.3]) == 0.7


def test_classifier_thresholds_raise_but_do_not_lower_risk() -> None:
    dangerous = analyzers._apply_phishing_guard(
        SAFE_PREDICTION,
        0.91,
        phishing_threshold=0.75,
        suspicious_threshold=0.50,
    )
    suspicious = analyzers._apply_phishing_guard(
        SAFE_PREDICTION,
        0.62,
        phishing_threshold=0.75,
        suspicious_threshold=0.50,
    )
    already_dangerous = dict(SAFE_PREDICTION, risk_level="dangerous")
    unchanged = analyzers._apply_phishing_guard(
        already_dangerous,
        0.1,
        phishing_threshold=0.75,
        suspicious_threshold=0.50,
    )

    assert dangerous["risk_level"] == "dangerous"
    assert dangerous["scam_type"] == "phishing"
    assert "Do not click links" in dangerous["safest_action"]
    assert suspicious["risk_level"] == "suspicious"
    assert unchanged["risk_level"] == "dangerous"


def test_hybrid_builder_passes_classifier_signal_as_system_context(monkeypatch) -> None:
    contexts = []

    def fake_llama_builder(model_path, *, system_context_provider, **kwargs):
        def explain(message):
            contexts.append(system_context_provider(message))
            return dict(SAFE_PREDICTION)

        return explain

    monkeypatch.setattr(analyzers, "build_llama_cpp_analyzer", fake_llama_builder)
    hybrid = analyzers.build_hybrid_analyzer(
        Path("local-model.gguf"),
        classifier=lambda _: 0.9,
    )

    prediction = hybrid("This email asks me to verify my account.")

    assert "combined phishing score is 0.900" in contexts[0]
    assert prediction["risk_level"] == "dangerous"


def test_app_dispatches_hybrid_backend_without_changing_default(monkeypatch, tmp_path) -> None:
    app.get_analyzer.cache_clear()
    monkeypatch.setenv("JAWBREAKER_BACKEND", "hybrid")
    monkeypatch.setenv("JAWBREAKER_CLASSIFIER_MODEL_ID", "local/test-classifier")
    monkeypatch.setenv("JAWBREAKER_PHISHING_THRESHOLD", "0.8")
    monkeypatch.setenv("JAWBREAKER_SUSPICIOUS_THRESHOLD", "0.4")
    monkeypatch.setattr(app, "resolve_model_path", lambda: tmp_path / "explainer.gguf")
    calls = {}

    def fake_builder(model_path, **kwargs):
        calls.update(model_path=model_path, **kwargs)
        return lambda message: dict(SAFE_PREDICTION)

    monkeypatch.setattr(app, "build_hybrid_analyzer", fake_builder)

    try:
        assert app.get_analyzer()("sample") ["risk_level"] == "safe"
        assert calls["model_path"] == tmp_path / "explainer.gguf"
        assert calls["classifier_model_id"] == "local/test-classifier"
        assert calls["phishing_threshold"] == 0.8
        assert calls["suspicious_threshold"] == 0.4
    finally:
        app.get_analyzer.cache_clear()

    monkeypatch.setenv("JAWBREAKER_BACKEND", "llama-cpp")
    assert app.current_backend() == "llama-cpp"


def test_hybrid_model_status_label_is_explicit(monkeypatch) -> None:
    monkeypatch.setenv("JAWBREAKER_BACKEND", "hybrid")

    assert app.model_status_label() == "Hybrid (DistilBERT + MiniCPM5-1B)"
    assert app.backend_status_label() == "LOCAL HYBRID"
