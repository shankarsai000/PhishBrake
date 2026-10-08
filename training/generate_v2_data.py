from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.generate_jawbreaker_data import (  # noqa: E402
    eval_row,
    generated_cases,
    sft_row,
    write_jsonl,
)


HARD_CASES = [
    {
        "category": "family_impersonation_hard",
        "risk": "dangerous",
        "scam_type": "family_impersonation",
        "tactics": ["impersonation", "urgency", "new channel", "payment pressure"],
        "impersonates": "family member",
        "pressure": "new number and deadline",
        "ask": "send money",
        "risk_text": "payment theft",
        "safe_action": "Do not send money. Call the person using a number you already know or ask another family member to verify.",
        "messages": [
            "Grandpa, my phone broke. Text me on this number only. I need help paying a bill before midnight.",
            "This is your son using a friend's phone. I am stranded. Can you send crypto to this wallet right away?",
            "Mom it's me from a temporary phone. I need help with rent today and cannot talk.",
            "Grandma, please use this new number. My account is locked and I need you to send money now.",
        ],
    },
    {
        "category": "secrecy_payment_hard",
        "risk": "dangerous",
        "scam_type": "payment_request",
        "tactics": ["secrecy", "payment pressure", "emotional manipulation"],
        "impersonates": "trusted person",
        "pressure": "keep it secret",
        "ask": "buy gift cards or pay a bill",
        "risk_text": "money theft",
        "safe_action": "Do not send money or buy gift cards. Verify with the person through a trusted route first.",
        "messages": [
            "Auntie please don't tell Dad. I need gift cards for an emergency and I'll pay you back tomorrow.",
            "Do not tell your family about us yet. I need help with a hospital bill before we can meet.",
            "Please keep this quiet. I need Apple gift cards today and I will explain later.",
            "Nobody else can know yet. Wire the hospital payment and I will repay you tomorrow.",
        ],
    },
    {
        "category": "marketplace_hard",
        "risk": "suspicious",
        "scam_type": "marketplace_scam",
        "tactics": ["marketplace", "payment pressure", "third party"],
        "impersonates": "buyer",
        "pressure": "unusual payment arrangement",
        "ask": "use escrow, courier, code, or bank detail",
        "risk_text": "overpayment or account theft",
        "safe_action": "Pause before acting. Do not send codes, bank details, or upfront payments; use the marketplace's official payment flow.",
        "messages": [
            "I want to buy your item. I will send extra money and my mover will pick it up.",
            "I only use this escrow service for marketplace purchases: https://safe-escrow-pay.example",
            "The courier needs a small insurance payment before pickup. I will reimburse you.",
            "I am out of town but my assistant will handle payment. Please email your bank name.",
            "Can you verify you are real by sending me the code I just texted you?",
        ],
    },
    {
        "category": "suspicious_link_hard",
        "risk": "suspicious",
        "scam_type": "unknown_link",
        "tactics": ["suspicious link", "vague request"],
        "impersonates": "unknown sender",
        "pressure": "vague account or document issue",
        "ask": "open a link",
        "risk_text": "possible credential theft",
        "safe_action": "Do not use the link from the message. Open the official website or app yourself if you need to check.",
        "messages": [
            "Your account needs attention. Visit http://account-message-center.example when you have time.",
            "We could not process your request. Open https://customer-resolution.example for details.",
            "Your profile is incomplete. Confirm your information here: http://profile-update-center.example",
            "Please review the secure document I shared: https://document-viewer-secure.example",
            "Your voicemail is available at http://voice-message-download.example",
        ],
    },
    {
        "category": "suspicious_urgency_hard",
        "risk": "suspicious",
        "scam_type": "unknown_urgent_request",
        "tactics": ["urgency", "vague request"],
        "impersonates": "unknown sender",
        "pressure": "act quickly",
        "ask": "confirm or respond",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Pause before responding. Verify through an official app, website, or known phone number.",
        "messages": [
            "Please call me back immediately about your account. This cannot wait.",
            "Your subscription has a problem. We need confirmation today or it may be closed.",
            "Final reminder: your paperwork is missing. Send your details by end of day.",
            "Action required: confirm ownership of your account within 24 hours.",
            "We noticed an issue with your delivery. Respond quickly so we can resolve it.",
        ],
    },
    {
        "category": "legitimate_notice_hard",
        "risk": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "tactics": ["verification needed", "trusted route"],
        "impersonates": "service provider",
        "pressure": "routine account notice",
        "ask": "verify through official route",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Verify directly through the official app, official website, or a known phone number.",
        "messages": [
            "UPS update: your package is out for delivery today. Track it in the UPS app.",
            "FedEx delivery exception: weather delay reported. Check status on the official FedEx site.",
            "Your pharmacy order shipped. Open your pharmacy account if you need details.",
            "Your password was changed. If this was not you, open the app directly and review account security.",
            "Your verification code is 482910. Do not share this code with anyone.",
            "School reminder: permission forms are due Friday. Check the parent portal for details.",
            "Your utility bill is ready. No payment is due until next week. View it through your normal account.",
        ],
    },
    {
        "category": "safe_hard",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": "known sender",
        "pressure": "no pressure",
        "ask": "no sensitive ask",
        "risk_text": "none identified",
        "safe_action": "If this came from someone you know and asks for nothing sensitive, it is probably safe. Still avoid unexpected links.",
        "messages": [
            "Reminder: dental appointment tomorrow at 2 PM. Call the office number you already have if you need to reschedule.",
            "Your lab results are available in the patient portal. Log in through the clinic website.",
            "School reminder: permission forms are due Friday. Check the parent portal for details.",
            "Thanks for your purchase. Your receipt is available in your account.",
            "Can you bring milk on the way home?",
        ],
    },
]


def summary_for(case: dict) -> str:
    risk = case["risk"]
    if risk == "dangerous":
        return f"This looks dangerous: likely {case['scam_type'].replace('_', ' ')}."
    if risk == "suspicious":
        return "This has warning signs and should be checked before acting."
    if risk == "needs_check":
        return "This might be legitimate, but it should be verified through a trusted route."
    return "No strong scam pattern was found in this short scan."


def trusted_message_for(case: dict) -> str:
    if case["risk"] == "safe":
        return "Can you sanity-check this message for me? Jawbreaker did not find a strong scam pattern, but I want to be careful."
    return (
        "Can you check this message with me before I do anything? "
        f"Jawbreaker marked it as {case['risk']} and says the safest next step is: {case['safe_action']}"
    )


def hard_prediction(case: dict) -> dict:
    return {
        "risk_level": case["risk"],
        "scam_type": case["scam_type"],
        "tactics": case["tactics"],
        "scam_dna": {
            "impersonates": case["impersonates"],
            "pressure": case["pressure"],
            "ask": case["ask"],
            "risk": case["risk_text"],
        },
        "safest_action": case["safe_action"],
        "trusted_person_message": trusted_message_for(case),
        "summary": summary_for(case),
    }


def hard_cases(repeats: int) -> list[tuple[str, dict, dict]]:
    cases = []
    for repeat in range(repeats):
        for case in HARD_CASES:
            prediction = hard_prediction(case)
            for index, message in enumerate(case["messages"]):
                case_id = f"{case['category']}_{repeat:02d}_{index:02d}"
                cases.append((case_id, {"message": message, "scenario": case}, prediction))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jawbreaker v2 hard-case training data.")
    parser.add_argument("--base-train", type=int, default=720)
    parser.add_argument("--base-dev", type=int, default=120)
    parser.add_argument("--base-test", type=int, default=180)
    parser.add_argument("--hard-repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "eval" / "hard_v2_eval.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.base_train + args.base_dev + args.base_test
    base = list(generated_cases(total))
    hard = hard_cases(args.hard_repeats)

    split = max(1, len(hard) // 5)
    hard_dev = hard[:split]
    hard_train = hard[split:]

    train_cases = base[: args.base_train] + hard_train
    dev_cases = base[args.base_train : args.base_train + args.base_dev] + hard_dev
    test_cases = base[args.base_train + args.base_dev :] + hard_dev

    write_jsonl(args.out_dir / "train_v2.jsonl", (sft_row(*case) for case in train_cases))
    write_jsonl(args.out_dir / "dev_v2.jsonl", (sft_row(*case) for case in dev_cases))
    write_jsonl(args.out_dir / "test_v2.jsonl", (sft_row(*case) for case in test_cases))
    write_jsonl(args.eval_out, (eval_row(*case) for case in test_cases))

    print(f"wrote train_v2={len(train_cases)} dev_v2={len(dev_cases)} test_v2={len(test_cases)}")
    print(f"hard_cases={len(hard)} hard_train={len(hard_train)} hard_dev={len(hard_dev)}")
    print(f"training data: {args.out_dir}")
    print(f"hard eval: {args.eval_out}")


if __name__ == "__main__":
    main()
