import json
from pathlib import Path

from jawbreaker.schema import RISK_LEVELS


def test_eval_dataset_has_100_unique_cases() -> None:
    rows = [
        json.loads(line)
        for line in Path("eval/scam_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100


def test_eval_dataset_required_fields_and_risk_levels() -> None:
    required = {"id", "category", "input", "expected_risk_level", "expected_scam_type", "expected_tactics"}
    rows = [
        json.loads(line)
        for line in Path("eval/scam_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for row in rows:
        assert required <= set(row)
        assert row["expected_risk_level"] in RISK_LEVELS
        assert isinstance(row["expected_tactics"], list)
        assert ".example" in row["input"] or "http" not in row["input"].lower()


def test_generated_training_splits_exist_and_do_not_overlap() -> None:
    paths = {
        "train": Path("training/data/train.jsonl"),
        "dev": Path("training/data/dev.jsonl"),
        "test": Path("training/data/test.jsonl"),
    }
    expected_counts = {"train": 720, "dev": 120, "test": 180}
    seen_messages = set()

    for split, path in paths.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == expected_counts[split]
        for row in rows:
            assert {"id", "messages", "input", "prediction"} <= set(row)
            assert row["input"] not in seen_messages
            seen_messages.add(row["input"])
            prediction = row["prediction"]
            assert prediction["risk_level"] in RISK_LEVELS
            assert set(prediction["scam_dna"]) == {"impersonates", "pressure", "ask", "risk"}
            assert row["messages"][-1]["role"] == "assistant"


def test_generated_eval_has_test_count_and_safe_urls() -> None:
    rows = [
        json.loads(line)
        for line in Path("eval/generated_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 180
    assert len({row["id"] for row in rows}) == 180
    for row in rows:
        assert row["expected_risk_level"] in RISK_LEVELS
        assert ".example" in row["input"] or "http" not in row["input"].lower()


def test_field_examples_are_sanitized_and_valid() -> None:
    rows = [
        json.loads(line)
        for line in Path("eval/field_examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) >= 2
    for row in rows:
        assert {"id", "category", "input", "expected_risk_level", "expected_scam_type", "expected_tactics"} <= set(row)
        assert row["expected_risk_level"] in RISK_LEVELS
        assert "[phone number]" in row["input"] or "[callback number]" in row["input"]
        assert "Vineel" not in row["input"]
        assert "+1" not in row["input"]


def test_fresh_2026_eval_is_sanitized_and_balanced() -> None:
    rows = [
        json.loads(line)
        for line in Path("eval/fresh_2026_scam_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100

    risk_counts = {}
    categories = set()
    for row in rows:
        assert {"id", "category", "input", "expected_risk_level", "expected_scam_type", "expected_tactics"} <= set(row)
        assert row["expected_risk_level"] in RISK_LEVELS
        assert isinstance(row["expected_tactics"], list)
        assert ".example" in row["input"] or "http" not in row["input"].lower()
        assert "+1" not in row["input"]
        assert "@" not in row["input"]
        risk_counts[row["expected_risk_level"]] = risk_counts.get(row["expected_risk_level"], 0) + 1
        categories.add(row["category"])

    assert risk_counts == {"dangerous": 72, "needs_check": 16, "safe": 12}
    assert {
        "toll_smishing",
        "package_phishing",
        "callback_phishing",
        "job_scam",
        "investment_scam",
        "credential_theft",
        "government_impersonation",
        "marketplace_scam",
        "tech_support",
        "legitimate_needs_check",
        "safe_benign",
    } <= categories


def test_v7_training_data_is_sanitized_and_separate_from_fresh_eval() -> None:
    split_paths = {
        "train": Path("training/data/train_v7.jsonl"),
        "dev": Path("training/data/dev_v7.jsonl"),
        "test": Path("training/data/test_v7.jsonl"),
    }
    expected_counts = {"train": 2192, "dev": 498, "test": 558}
    fresh_eval_inputs = {
        json.loads(line)["input"]
        for line in Path("eval/fresh_2026_scam_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    for split, path in split_paths.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == expected_counts[split]
        assert not ({row["input"] for row in rows} & fresh_eval_inputs)

        for row in rows:
            assert {"id", "messages", "input", "prediction"} <= set(row)
            assert row["prediction"]["risk_level"] in RISK_LEVELS
            assert ".example" in row["input"] or "http" not in row["input"].lower()
            assert "+1" not in row["input"]
            assert "@" not in row["input"]

    hard_rows = [
        json.loads(line)
        for line in Path("eval/hard_v7_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(hard_rows) == expected_counts["test"]
    assert {
        "public_pattern_wrong_number_crypto_v7",
        "public_pattern_marketplace_money_v7",
        "public_pattern_task_job_v7",
        "public_pattern_mfa_code_v7",
        "public_pattern_toll_tax_benefit_v7",
        "safe_everyday_family_v7",
        "needs_check_official_route_v7",
    } <= {row["category"] for row in hard_rows}


def test_v8_training_data_is_sanitized_and_separate_from_fresh_eval() -> None:
    split_paths = {
        "train": Path("training/data/train_v8.jsonl"),
        "dev": Path("training/data/dev_v8.jsonl"),
        "test": Path("training/data/test_v8.jsonl"),
    }
    expected_counts = {"train": 2488, "dev": 572, "test": 632}
    fresh_eval_inputs = {
        json.loads(line)["input"]
        for line in Path("eval/fresh_2026_scam_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    for split, path in split_paths.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == expected_counts[split]
        assert not ({row["input"] for row in rows} & fresh_eval_inputs)

        for row in rows:
            assert {"id", "messages", "input", "prediction"} <= set(row)
            assert row["prediction"]["risk_level"] in RISK_LEVELS
            assert ".example" in row["input"] or "http" not in row["input"].lower()
            assert "+1" not in row["input"]
            assert "@" not in row["input"]

    hard_rows = [
        json.loads(line)
        for line in Path("eval/hard_v8_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(hard_rows) == expected_counts["test"]
    assert {
        "wrong_number_investment_danger_v8",
        "wrong_number_social_no_money_v8",
        "safe_family_logistics_v8",
        "safe_school_pickup_v8",
        "safe_pharmacy_clinic_v8",
        "school_clinic_payment_link_danger_v8",
        "official_route_needs_check_v8",
    } <= {row["category"] for row in hard_rows}
