import importlib.util
from pathlib import Path

from jawbreaker.analyzers import load_json_prediction, validate_prediction


def load_run_eval_module():
    spec = importlib.util.spec_from_file_location("run_eval", Path("eval/run_eval.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_prediction_accepts_complete_prediction() -> None:
    prediction = {
        "risk_level": "dangerous",
        "scam_type": "credential_theft",
        "summary": "This is pretending to be a bank.",
        "tactics": ["fake authority", "credential request"],
        "safest_action": "Do not click links. Open the official app directly.",
        "trusted_person_message": "Can you check this for me?",
        "scam_dna": {
            "impersonates": "bank",
            "pressure": "account locked",
            "ask": "login",
            "risk": "credential theft",
        },
    }

    assert validate_prediction(prediction) == []


def test_score_rows_tracks_dangerous_as_safe() -> None:
    run_eval = load_run_eval_module()
    rows = [
        {
            "id": "case_1",
            "category": "bank_phishing",
            "input": "Bank login now",
            "expected_risk_level": "dangerous",
            "expected_scam_type": "credential_theft",
            "expected_tactics": ["credential request"],
        }
    ]
    predictions = {
        "case_1": {
            "risk_level": "safe",
            "scam_type": "none",
            "summary": "Looks fine.",
            "tactics": [],
            "safest_action": "No action needed.",
            "trusted_person_message": "Can you check this?",
            "scam_dna": {"impersonates": "", "pressure": "", "ask": "", "risk": ""},
        }
    }

    metrics = run_eval.score_rows(rows, predictions, elapsed=0.01)

    assert metrics["risk_level_accuracy"] == 0
    assert metrics["dangerous_as_safe"] == ["case_1"]
    assert metrics["dangerous_as_needs_check"] == []
    assert metrics["suspicious_as_safe"] == []


def test_score_rows_tracks_dangerous_undercalls_and_suspicious_as_safe() -> None:
    run_eval = load_run_eval_module()
    rows = [
        {
            "id": "danger_case",
            "category": "family_impersonation",
            "input": "Grandpa, I need money before midnight.",
            "expected_risk_level": "dangerous",
            "expected_scam_type": "family_impersonation",
            "expected_tactics": ["payment pressure"],
        },
        {
            "id": "suspicious_case",
            "category": "suspicious",
            "input": "Open this marketplace escrow link.",
            "expected_risk_level": "suspicious",
            "expected_scam_type": "fake_escrow",
            "expected_tactics": ["suspicious link"],
        },
    ]
    base_prediction = {
        "scam_type": "unknown",
        "summary": "Check this.",
        "tactics": [],
        "safest_action": "Verify through a trusted route.",
        "trusted_person_message": "Can you check this?",
        "scam_dna": {"impersonates": "", "pressure": "", "ask": "", "risk": ""},
    }
    predictions = {
        "danger_case": {**base_prediction, "risk_level": "needs_check"},
        "suspicious_case": {**base_prediction, "risk_level": "safe"},
    }

    metrics = run_eval.score_rows(rows, predictions, elapsed=0.01)

    assert metrics["dangerous_as_needs_check"] == ["danger_case"]
    assert metrics["suspicious_as_safe"] == ["suspicious_case"]


def test_score_rows_tracks_model_errors() -> None:
    run_eval = load_run_eval_module()
    rows = [
        {
            "id": "case_1",
            "category": "safe_benign",
            "input": "Dentist appointment Tuesday.",
            "expected_risk_level": "safe",
            "expected_scam_type": "none",
            "expected_tactics": [],
        }
    ]
    predictions = {
        "case_1": {
            "risk_level": "safe",
            "scam_type": "none",
            "summary": "Looks fine.",
            "tactics": [],
            "safest_action": "No action needed.",
            "trusted_person_message": "Can you check this?",
            "scam_dna": {"impersonates": "", "pressure": "", "ask": "", "risk": ""},
            "_jawbreaker_model_error": "JSONDecodeError('empty')",
        }
    }

    metrics = run_eval.score_rows(rows, predictions, elapsed=0.01)

    assert metrics["model_errors"] == [{"id": "case_1", "error": "JSONDecodeError('empty')"}]


def test_has_unsafe_action_allows_do_not_send_money() -> None:
    run_eval = load_run_eval_module()

    assert not run_eval.has_unsafe_action("Do not send money. Call a known number.")
    assert run_eval.has_unsafe_action("Send money to verify the account.")


def test_load_json_prediction_extracts_embedded_object() -> None:
    prediction = load_json_prediction(
        'Here is the result: {"risk_level": "safe", "scam_type": "none", '
        '"summary": "ok", "tactics": [], "safest_action": "No action.", '
        '"trusted_person_message": "Please check.", '
        '"scam_dna": {"impersonates": "", "pressure": "", "ask": "", "risk": ""}}'
    )

    assert prediction["risk_level"] == "safe"


def test_load_json_prediction_ignores_qwen_thinking_tokens() -> None:
    prediction = load_json_prediction(
        '<think>I should reason internally.</think>{"risk_level": "dangerous", '
        '"scam_type": "family_impersonation", "summary": "scam", '
        '"tactics": ["secrecy"], "safest_action": "Do not reply.", '
        '"trusted_person_message": "Can you check this?", '
        '"scam_dna": {"impersonates": "family", "pressure": "secret", "ask": "money", "risk": "payment"}}'
    )

    assert prediction["risk_level"] == "dangerous"
