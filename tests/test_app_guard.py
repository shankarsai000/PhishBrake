from app import (
    DEFAULT_ADAPTER_ID,
    DEFAULT_TRANSFORMERS_MAX_TOKENS,
    DEFAULT_TRANSFORMERS_MODEL_ID,
    build_handoff_message,
    default_adapter_id,
    model_status_label,
    remember_current,
    resolve_model_path,
    run_analysis,
    should_use_heuristic_guard,
)
from jawbreaker.analyzers import load_json_prediction, repair_prediction
from jawbreaker.schema import ScamAnalysis


def test_resolve_model_path_defaults_to_minicpm_gguf(monkeypatch, tmp_path) -> None:
    import huggingface_hub

    monkeypatch.delenv("JAWBREAKER_MODEL_PATH", raising=False)
    monkeypatch.delenv("JAWBREAKER_MODEL_REPO", raising=False)
    monkeypatch.delenv("JAWBREAKER_MODEL_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    downloaded_path = tmp_path / "minicpm5-1b-Q4_K_M.gguf"
    calls = {}

    def fake_download(*, repo_id, filename, cache_dir=None):
        calls.update(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        return str(downloaded_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert resolve_model_path() == downloaded_path
    assert calls["repo_id"] == "Abiray/MiniCPM5-1B-GGUF"
    assert calls["filename"] == "minicpm5-1b-Q4_K_M.gguf"


def test_guard_uses_heuristic_when_model_undercalls_dangerous_message() -> None:
    model = ScamAnalysis(
        risk_level="needs_check",
        scam_type="unknown",
        summary="Check this before acting.",
        tactics=[],
        scam_dna={"Impersonates": "", "Pressure": "", "Ask": "", "Risk": ""},
    )
    heuristic = ScamAnalysis.from_heuristics(
        "Hi Grandma, I lost my phone. This is my new number. Can you send $800 today?"
    )

    assert should_use_heuristic_guard(
        model,
        heuristic,
        validation_errors=[],
        message="Hi Grandma, I lost my phone. This is my new number. Can you send $800 today?",
    )


def test_guard_keeps_informative_model_result() -> None:
    model = ScamAnalysis(
        risk_level="dangerous",
        scam_type="family_impersonation",
        summary="This looks like a family emergency scam.",
        tactics=["impersonation", "urgency"],
        scam_dna={
            "Impersonates": "Family member",
            "Pressure": "Act today",
            "Ask": "Send money",
            "Risk": "payment request",
        },
    )
    heuristic = ScamAnalysis.from_heuristics(
        "Hi Grandma, I lost my phone. This is my new number. Can you send $800 today?"
    )

    assert not should_use_heuristic_guard(model, heuristic, validation_errors=[])


def test_guard_does_not_escalate_official_route_notice_to_dangerous() -> None:
    message = "Credit union alert: we noticed a new login. If this was not you, open the official app."
    model = ScamAnalysis(
        risk_level="needs_check",
        scam_type="possible_legitimate_alert",
        summary="Verify this through the official app.",
        tactics=["verification needed"],
        scam_dna={
            "Impersonates": "credit union",
            "Pressure": "routine security notice",
            "Ask": "open official app",
            "Risk": "uncertain legitimacy",
        },
    )
    heuristic = ScamAnalysis.from_heuristics(message)

    assert not should_use_heuristic_guard(model, heuristic, validation_errors=[], message=message)


def test_guard_does_not_escalate_password_changed_official_app_notice() -> None:
    message = "Coinbase notice: your password was changed. Open the official app yourself to review account security."
    model = ScamAnalysis(
        risk_level="needs_check",
        scam_type="possible_legitimate_alert",
        summary="Verify this through the official app.",
        tactics=["verification needed"],
        scam_dna={
            "Impersonates": "Coinbase",
            "Pressure": "routine security notice",
            "Ask": "open official app",
            "Risk": "uncertain legitimacy",
        },
    )
    heuristic = ScamAnalysis.from_heuristics(message)

    assert heuristic.risk_level == "dangerous"
    assert not should_use_heuristic_guard(model, heuristic, validation_errors=[], message=message)


def test_guard_downgrades_unrequested_security_code_model_false_positive() -> None:
    message = (
        "Your GitHub security code is 849201. It expires in 10 minutes. "
        "If you did not request this code, no action is required."
    )
    heuristic = ScamAnalysis.from_heuristics(message)
    model = ScamAnalysis(
        risk_level="dangerous",
        scam_type="code_scaver",
        summary="This message is a warning sign of code theft.",
        tactics=["code request"],
        scam_dna={"Impersonates": "GitHub", "Pressure": "expiry", "Ask": "code", "Risk": "account"},
    )

    assert heuristic.risk_level == "safe"
    assert should_use_heuristic_guard(model, heuristic, validation_errors=[], message=message)

    uncertain_model = ScamAnalysis(
        risk_level="needs_check",
        scam_type="unknown",
        summary="Verify this message.",
        tactics=[],
        scam_dna={"Impersonates": "", "Pressure": "", "Ask": "", "Risk": ""},
    )
    assert should_use_heuristic_guard(
        uncertain_model,
        heuristic,
        validation_errors=[],
        message=message,
    )


def test_guard_does_not_exempt_requested_security_code() -> None:
    message = (
        "Your GitHub security code is 849201. If you did not request this code, no action is required. "
        "Reply with this code to keep access."
    )
    heuristic = ScamAnalysis.from_heuristics(message)
    model = ScamAnalysis(
        risk_level="dangerous",
        scam_type="code_theft",
        summary="The sender asks you to share a code.",
        tactics=["credential request"],
        scam_dna={"Impersonates": "GitHub", "Pressure": "access", "Ask": "code", "Risk": "account"},
    )

    assert not should_use_heuristic_guard(model, heuristic, validation_errors=[], message=message)


def test_guard_promotes_wrong_number_investment_undercall() -> None:
    message = "Wrong chat. You are polite, so I can invite you to a quiet investment group before the window closes."
    model = ScamAnalysis(
        risk_level="needs_check",
        scam_type="unknown_contact",
        summary="This should be checked.",
        tactics=["wrong number"],
        scam_dna={
            "Impersonates": "unknown sender",
            "Pressure": "friendly contact",
            "Ask": "continue chatting",
            "Risk": "uncertain legitimacy",
        },
    )
    heuristic = ScamAnalysis.from_heuristics(message)

    assert heuristic.risk_level == "dangerous"
    assert heuristic.scam_type == "investment_scam"
    assert should_use_heuristic_guard(model, heuristic, validation_errors=[], message=message)


def test_guard_promotes_indirect_wrong_number_investment_language() -> None:
    message = "I thought this was Lena. My aunt teaches a gold strategy and can reserve you a spot if you open the app."
    model = ScamAnalysis(
        risk_level="suspicious",
        scam_type="unknown_contact",
        summary="This has warning signs.",
        tactics=["relationship building"],
        scam_dna={
            "Impersonates": "unknown sender",
            "Pressure": "friendly contact",
            "Ask": "open app",
            "Risk": "uncertain legitimacy",
        },
    )
    heuristic = ScamAnalysis.from_heuristics(message)

    assert heuristic.risk_level == "dangerous"
    assert heuristic.scam_type == "investment_scam"
    assert should_use_heuristic_guard(model, heuristic, validation_errors=[], message=message)


def test_guard_keeps_plain_wrong_number_social_contact_suspicious() -> None:
    message = "Wrong number, but thanks for being kind."
    model = ScamAnalysis(
        risk_level="safe",
        scam_type="none",
        summary="No scam pattern found.",
        tactics=[],
        scam_dna={"Impersonates": "", "Pressure": "", "Ask": "", "Risk": ""},
    )
    heuristic = ScamAnalysis.from_heuristics(message)

    assert heuristic.risk_level == "suspicious"
    assert should_use_heuristic_guard(model, heuristic, validation_errors=[], message=message)


def test_guard_demotes_routine_school_notice_false_alarm() -> None:
    message = "Office reminder: lunch account notices go home in backpacks today."
    model = ScamAnalysis(
        risk_level="suspicious",
        scam_type="unknown",
        summary="This might need checking.",
        tactics=["urgency"],
        scam_dna={
            "Impersonates": "school office",
            "Pressure": "today",
            "Ask": "read notice",
            "Risk": "uncertain legitimacy",
        },
    )
    heuristic = ScamAnalysis.from_heuristics(message)

    assert heuristic.risk_level == "safe"
    assert should_use_heuristic_guard(model, heuristic, validation_errors=[], message=message)


def test_eval_guard_keeps_official_route_notice_calibrated() -> None:
    from eval.run_eval import apply_safety_guard

    message = "Hospital portal: new test results are available. Sign in through your usual patient portal."
    prediction = {
        "risk_level": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "summary": "Verify this through the normal patient portal.",
        "tactics": ["verification needed"],
        "safest_action": "Verify directly through the official app, official website, or a known phone number.",
        "trusted_person_message": "Can you check this with me?",
        "scam_dna": {
            "impersonates": "hospital portal",
            "pressure": "routine notice",
            "ask": "use normal patient portal",
            "risk": "uncertain legitimacy",
        },
    }

    guarded, promoted = apply_safety_guard(message, prediction)

    assert not promoted
    assert guarded["risk_level"] == "needs_check"


def test_remember_current_saves_last_scan_without_reanalysis(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("remember_current should not run analysis")

    monkeypatch.setattr("app.run_analysis", fail_if_called)

    status, memory_html, memory = remember_current(
        "Hi Grandma, I lost my phone.",
        [],
        {
            "message": "Hi Grandma, I lost my phone.",
            "entry": {
                "summary": "This looks like family impersonation.",
                "scam_type": "family_impersonation",
                "risk_level": "dangerous",
                "fingerprint": {"Impersonates": "Family member"},
                "text": "Hi Grandma, I lost my phone.",
            },
        },
    )

    assert "Saved this scam pattern for this session." in status
    assert len(memory) == 1
    assert "Session scam memory" in memory_html


def test_handoff_message_includes_original_text_and_safe_action() -> None:
    message = "Hi Grandma, I lost my phone. This is my new number. Can you send $800 today?"
    analysis = ScamAnalysis.from_heuristics(message)

    handoff = build_handoff_message(message, analysis)

    assert "Message I received:" in handoff
    assert message in handoff
    assert "PhishBrake marked it as dangerous (family impersonation)." in handoff
    assert "Safest next step:" in handoff
    assert analysis.safest_action in handoff
    assert "I have not clicked any links, replied, sent money, or shared codes." in handoff


def test_run_analysis_falls_back_to_heuristic_when_model_fails(monkeypatch) -> None:
    def broken_analyzer():
        def analyze(message: str):
            raise ValueError("model returned invalid JSON")

        return analyze

    monkeypatch.setattr("app.get_analyzer", broken_analyzer)
    analysis = run_analysis(
        "Hi Grandma, I lost my phone. This is my new number. Can you send $800 today?",
        [],
    )

    assert analysis.risk_level == "dangerous"
    assert analysis.scam_type == "family_impersonation"
    assert "safety fallback" in analysis.summary


def test_run_analysis_keeps_typosquat_dangerous_when_model_fails(monkeypatch) -> None:
    def broken_analyzer():
        def analyze(message: str):
            raise RuntimeError("model unavailable")

        return analyze

    monkeypatch.setattr("app.get_analyzer", broken_analyzer)

    analysis = run_analysis("Please see https://paypa1.com for details.", [])

    assert analysis.risk_level == "dangerous"
    assert "Do not click links" in analysis.safest_action


def test_repair_prediction_adds_missing_summary_without_changing_risk() -> None:
    repaired = repair_prediction(
        {
            "risk_level": "dangerous",
            "scam_type": "family impersonation",
            "tactics": ["urgency"],
            "safest_action": "Do not send money.",
            "trusted_person_message": "Can you check this with me?",
            "scam_dna": {"impersonates": "family member"},
        }
    )

    assert repaired["risk_level"] == "dangerous"
    assert repaired["summary"] == "This looks dangerous: likely family impersonation."


def test_repair_prediction_sanitizes_unsafe_action() -> None:
    repaired = repair_prediction(
        {
            "risk_level": "dangerous",
            "scam_type": "family impersonation",
            "summary": "This is a money request.",
            "tactics": ["secrecy"],
            "safest_action": "Send gift cards today and explain later.",
            "trusted_person_message": "Can you check this with me?",
            "scam_dna": {"impersonates": "family member"},
        }
    )

    assert "Send gift cards" not in repaired["safest_action"]
    assert "Do not click links" in repaired["safest_action"]
    assert "do not send money" in repaired["safest_action"]


def test_repair_prediction_sanitizes_password_submission_action() -> None:
    repaired = repair_prediction(
        {
            "risk_level": "dangerous",
            "scam_type": "financial_security",
            "summary": "This is a phishing message.",
            "tactics": ["urgency"],
            "safest_action": "Verify at this link and enter your online banking password.",
            "trusted_person_message": "Can you check this with me?",
            "scam_dna": {"impersonates": "bank"},
        }
    )

    assert repaired["safest_action"].startswith("Do not click links")
    assert "enter your online banking password" not in repaired["safest_action"]


def test_load_json_prediction_recovers_near_json_model_output() -> None:
    prediction = load_json_prediction(
        """
        {"risk_level": "dangerous", "scam_type": "tech_support",
         "tactics": ["fake authority", "urgency"]
         "scam_dna": {"impersonates": "device support", "pressure": "avoid data loss",
         "ask": "call the alert number", "risk": "remote access theft"},
         "safest_action": "Do not call the number. Contact official support directly.",
         "trusted_person_message": "Can you check this with me?",
         "summary": "Fake device warning pushing a support call."}
        """
    )

    repaired = repair_prediction(prediction)

    assert repaired["risk_level"] == "dangerous"
    assert repaired["scam_type"] == "tech_support"
    assert repaired["tactics"] == ["fake authority", "urgency"]
    assert repaired["scam_dna"]["impersonates"] == "device support"
    assert repaired["summary"] == "Fake device warning pushing a support call."


def test_default_adapter_only_attaches_to_minicpm(monkeypatch) -> None:
    monkeypatch.delenv("JAWBREAKER_ADAPTER_ID", raising=False)

    assert default_adapter_id(DEFAULT_TRANSFORMERS_MODEL_ID) == DEFAULT_ADAPTER_ID
    assert default_adapter_id("Qwen/Qwen3-0.6B") is None
    assert DEFAULT_TRANSFORMERS_MAX_TOKENS == 512


def test_model_status_label_distinguishes_base_gguf_from_lora(monkeypatch) -> None:
    monkeypatch.setenv("JAWBREAKER_BACKEND", "llama-cpp")
    monkeypatch.delenv("JAWBREAKER_MODEL_FILE", raising=False)
    monkeypatch.delenv("JAWBREAKER_MODEL_PATH", raising=False)

    assert model_status_label() == "MiniCPM5-1B GGUF (base)"

    monkeypatch.setenv("JAWBREAKER_BACKEND", "transformers")
    assert model_status_label() == "MiniCPM5-1B + PhishBrake LoRA v8"
