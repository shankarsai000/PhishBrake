from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jawbreaker.contract import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402


SCENARIOS = [
    {
        "category": "package_phishing",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["fake authority", "urgency", "suspicious link", "credential request"],
        "impersonates": ["USPS", "FedEx", "UPS", "DHL", "Postal Service"],
        "pressure": ["held package", "final delivery attempt", "return to sender", "unpaid fee"],
        "ask": ["verify account details", "confirm address and payment", "update delivery login"],
        "risk_text": "credential theft",
        "safe_action": "Do not click the delivery link. Open the carrier website yourself or call a number you already trust.",
        "templates": [
            "{brand}: Your package is {pressure}. {ask_cap} now: https://{slug}.example",
            "{brand} notice: delivery failed. {ask_cap} before {time}: http://{slug}.example",
            "{brand}: {pressure_cap}. Your item will be returned unless you {ask}: https://{slug}.example",
        ],
    },
    {
        "category": "bank_phishing",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["fake authority", "urgency", "credential request"],
        "impersonates": ["Chase", "Bank of America", "credit union", "fraud department", "card security"],
        "pressure": ["account locked", "card suspended", "unusual transfer detected", "profile expires today"],
        "ask": ["send your password", "confirm your PIN", "reply with your one-time code", "verify your card number"],
        "risk_text": "credential theft",
        "safe_action": "Do not reply with codes, passwords, PINs, or card numbers. Open the official banking app or call the number on your card.",
        "templates": [
            "{brand}: your {pressure}. Please {ask} immediately.",
            "{brand} alert: {pressure_cap}. To cancel it, {ask}.",
            "Security notice from {brand}: {pressure_cap}. {ask_cap} now to restore access.",
        ],
    },
    {
        "category": "family_impersonation",
        "risk": "dangerous",
        "scam_type": "family_impersonation",
        "tactics": ["impersonation", "urgency", "secrecy", "payment pressure"],
        "impersonates": ["family member"],
        "pressure": ["emergency today", "new number", "do not tell anyone", "rent due today"],
        "ask": ["send money", "buy gift cards", "wire emergency funds", "send a payment app transfer"],
        "risk_text": "payment request",
        "safe_action": "Do not send money. Call the person using a number you already know or ask another family member to verify.",
        "templates": [
            "Hi {relative}, I lost my phone. This is my new number. Can you {ask} for {reason}? Please don't tell anyone.",
            "{relative}, it's me. I got in trouble and need help. {ask_cap} now and don't call anyone.",
            "Please keep this quiet. I need you to {ask} before {time}. I will explain later.",
        ],
    },
    {
        "category": "tech_support",
        "risk": "dangerous",
        "scam_type": "tech_support",
        "tactics": ["fake authority", "urgency", "remote access request"],
        "impersonates": ["Microsoft support", "Apple support", "security team", "antivirus support"],
        "pressure": ["device infected", "account compromised", "subscription expiring", "security warning"],
        "ask": ["install remote support software", "share the screen code", "call the number in this alert", "pay the cleanup fee"],
        "risk_text": "remote access or payment theft",
        "safe_action": "Do not call the number or install remote access tools. Close the message and contact official support through a trusted website.",
        "templates": [
            "{brand}: {pressure_cap}. {ask_cap} now to avoid permanent damage.",
            "Security alert: your {pressure}. Please {ask} immediately.",
            "{brand} case opened: {pressure_cap}. Technician requires you to {ask}.",
        ],
    },
    {
        "category": "prize_lottery",
        "risk": "dangerous",
        "scam_type": "prize_scam",
        "tactics": ["too good to be true", "payment request", "suspicious link"],
        "impersonates": ["lottery office", "reward center", "government grant office", "sweepstakes team"],
        "pressure": ["unclaimed prize", "reward expires tonight", "grant approved", "winner selected"],
        "ask": ["pay a processing fee", "enter your card number", "send your bank login", "claim at this link"],
        "risk_text": "money or credential theft",
        "safe_action": "Do not pay fees or share financial details to claim a prize. Verify through an official source you find yourself.",
        "templates": [
            "{brand}: {pressure_cap}. To receive it, {ask}: https://{slug}.example",
            "Congratulations, {pressure}. {ask_cap} before {time}.",
            "{brand} notice: {pressure_cap}. Release requires you to {ask}.",
        ],
    },
    {
        "category": "job_scam",
        "risk": "dangerous",
        "scam_type": "job_scam",
        "tactics": ["fake job", "payment request", "suspicious link"],
        "impersonates": ["recruiter", "hiring manager", "remote work office", "payroll team"],
        "pressure": ["job offer approved", "training starts today", "equipment shipment ready", "payroll setup pending"],
        "ask": ["pay an onboarding fee", "deposit a check and send back money", "buy equipment from this link", "share bank login"],
        "risk_text": "advance fee or check scam",
        "safe_action": "Do not pay to start a job or move money for an employer. Verify the company through its official website.",
        "templates": [
            "{brand}: {pressure_cap}. To continue, {ask}: https://{slug}.example",
            "You are hired. {ask_cap} before {time} so we can activate your role.",
            "{brand} here. Your {pressure}; please {ask} now.",
        ],
    },
    {
        "category": "legitimate_needs_check",
        "risk": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "tactics": ["verification needed"],
        "impersonates": ["bank or service provider"],
        "pressure": ["possible fraud alert", "appointment confirmation", "delivery update", "account notice"],
        "ask": ["confirm through official channel"],
        "risk_text": "uncertain legitimacy",
        "safe_action": "Verify directly through the official app, official website, or a known phone number.",
        "templates": [
            "{brand} fraud alert: Did you attempt a ${amount} purchase at {merchant}? Reply YES or NO.",
            "{brand}: Your appointment is confirmed for {time}. Reply C to confirm or R to reschedule.",
            "{brand}: We noticed a sign-in. If this was not you, open the official app to review.",
        ],
    },
    {
        "category": "safe_benign",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": ["known sender"],
        "pressure": ["no pressure"],
        "ask": ["no sensitive ask"],
        "risk_text": "none identified",
        "safe_action": "If this came from someone you know and asks for nothing sensitive, it is probably safe. Still avoid unexpected links.",
        "templates": [
            "Your dentist appointment is confirmed for {time}. Call the office number on your calendar if you need to reschedule.",
            "Can you bring milk when you come over tonight?",
            "The library book you requested is ready for pickup at the front desk.",
            "Dinner moved to {time}. No need to bring anything.",
        ],
    },
]


VALUES = {
    "time": ["2 PM", "6 PM", "tomorrow morning", "Friday", "tonight", "noon"],
    "relative": ["Grandma", "Mom", "Grandpa", "Auntie", "Dad"],
    "reason": ["rent today", "car repairs", "bail", "a hospital bill", "a school payment"],
    "amount": ["249.00", "82.19", "940.00", "1,200.00", "17.45"],
    "merchant": ["TARGET", "Walmart", "Shell", "Amazon", "Costco"],
}


def slugify(*parts: str) -> str:
    text = "-".join(parts).lower()
    return "".join(char if char.isalnum() else "-" for char in text).strip("-").replace("--", "-")


def first(values: list[str], index: int) -> str:
    return values[index % len(values)]


def build_prediction(scenario: dict, message: str, index: int) -> dict:
    return {
        "risk_level": scenario["risk"],
        "scam_type": scenario["scam_type"],
        "tactics": scenario["tactics"],
        "scam_dna": {
            "impersonates": first(scenario["impersonates"], index),
            "pressure": first(scenario["pressure"], index),
            "ask": first(scenario["ask"], index),
            "risk": scenario["risk_text"],
        },
        "safest_action": scenario["safe_action"],
        "trusted_person_message": trusted_message_for(scenario, message),
        "summary": summary_for(scenario),
    }


def summary_for(scenario: dict) -> str:
    if scenario["risk"] == "dangerous":
        return f"This looks dangerous: likely {scenario['scam_type'].replace('_', ' ')}."
    if scenario["risk"] == "suspicious":
        return "This has warning signs and should be checked before acting."
    if scenario["risk"] == "needs_check":
        return "This might be legitimate, but it should be verified through a trusted route."
    return "No strong scam pattern was found in this short scan."


def trusted_message_for(scenario: dict, message: str) -> str:
    if scenario["risk"] == "safe":
        return "Can you sanity-check this message for me? Jawbreaker did not find a strong scam pattern, but I want to be careful."
    return (
        "Can you check this message with me before I do anything? "
        f"Jawbreaker marked it as {scenario['risk']} and says the safest next step is: {scenario['safe_action']}"
    )


def render_message(scenario: dict, template: str, index: int) -> str:
    brand = first(scenario["impersonates"], index)
    pressure = first(scenario["pressure"], index)
    ask = first(scenario["ask"], index)
    slug = slugify(brand, pressure, ask, str(index))
    values = {
        "brand": brand,
        "pressure": pressure,
        "pressure_cap": pressure[:1].upper() + pressure[1:],
        "ask": ask,
        "ask_cap": ask[:1].upper() + ask[1:],
        "slug": slug,
        "time": first(VALUES["time"], index),
        "relative": first(VALUES["relative"], index),
        "reason": first(VALUES["reason"], index),
        "amount": first(VALUES["amount"], index),
        "merchant": first(VALUES["merchant"], index),
    }
    message = template.format(**values)
    if scenario["risk"] == "safe":
        return f"{message} Note {1000 + index}."
    return f"{message} Ref {1000 + index}."


def generated_cases(count: int) -> Iterable[tuple[str, dict, dict]]:
    for index in range(count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        template = scenario["templates"][(index // len(SCENARIOS)) % len(scenario["templates"])]
        message = render_message(scenario, template, index)
        prediction = build_prediction(scenario, message, index)
        case_id = f"{scenario['category']}_{index:04d}"
        yield case_id, {"message": message, "scenario": scenario}, prediction


def eval_row(case_id: str, item: dict, prediction: dict) -> dict:
    scenario = item["scenario"]
    return {
        "id": case_id,
        "category": scenario["category"],
        "input": item["message"],
        "expected_risk_level": prediction["risk_level"],
        "expected_scam_type": prediction["scam_type"],
        "expected_tactics": prediction["tactics"],
    }


def sft_row(case_id: str, item: dict, prediction: dict) -> dict:
    user_prompt = USER_PROMPT_TEMPLATE.format(message=item["message"])
    return {
        "id": case_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(prediction, ensure_ascii=True)},
        ],
        "input": item["message"],
        "prediction": prediction,
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jawbreaker synthetic train/eval data.")
    parser.add_argument("--train", type=int, default=720)
    parser.add_argument("--dev", type=int, default=120)
    parser.add_argument("--test", type=int, default=180)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "eval" / "generated_eval.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.train + args.dev + args.test
    cases = list(generated_cases(total))

    train_cases = cases[: args.train]
    dev_cases = cases[args.train : args.train + args.dev]
    test_cases = cases[args.train + args.dev :]

    write_jsonl(args.out_dir / "train.jsonl", (sft_row(*case) for case in train_cases))
    write_jsonl(args.out_dir / "dev.jsonl", (sft_row(*case) for case in dev_cases))
    write_jsonl(args.out_dir / "test.jsonl", (sft_row(*case) for case in test_cases))
    write_jsonl(args.eval_out, (eval_row(*case) for case in test_cases))

    print(f"wrote train={len(train_cases)} dev={len(dev_cases)} test={len(test_cases)}")
    print(f"training data: {args.out_dir}")
    print(f"generated eval: {args.eval_out}")


if __name__ == "__main__":
    main()
