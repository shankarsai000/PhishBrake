from __future__ import annotations

import argparse
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
from training.generate_v2_data import hard_cases, hard_prediction  # noqa: E402
from training.generate_v3_data import boundary_cases, split_special_cases  # noqa: E402
from training.generate_v4_data import v4_cases  # noqa: E402


CALIBRATION_CASES = [
    {
        "category": "official_route_calibration_v5",
        "risk": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "tactics": ["verification needed", "trusted route"],
        "impersonates": "service provider",
        "pressure": "routine safety or account notice",
        "ask": "verify through official app, official website, or known route",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Verify directly through the official app, official website, or a known phone number.",
        "messages": [
            "Bank notice: a new device signed in. If this was not you, open the official banking app and review security.",
            "Credit union alert: profile contact info changed. Use the mobile app or call the number on your card to review.",
            "Card alert: did you attempt a $44.18 grocery purchase? Reply YES or NO, or call the number on your card.",
            "Coinbase notice: your password was changed. Open the official app yourself to review account security.",
            "PayPal security: a new passkey was added. Go through the app or paypal.com directly if this was not you.",
            "Utility account: your statement is ready. Pay only through your normal account or the number printed on your bill.",
            "Phone carrier: SIM protection was updated. Open the carrier app directly to check your account.",
            "County office: your property notice is ready. Visit the official county website from prior mail.",
            "Airline alert: your flight changed. Open the airline app directly to confirm the new time.",
            "Insurance notice: explanation of benefits is ready. Review it through your normal member portal.",
            "School portal: report cards are available. Sign in through the normal parent portal.",
            "Delivery update: your package is out for delivery. Track it in the official carrier app.",
            "Pharmacy: prescription refill is ready. Use your normal pharmacy account or store phone number for details.",
            "Hospital portal: new lab results are available. Sign in through your usual patient portal.",
            "Streaming service: payment method needs attention. Open the official app or website yourself.",
            "Library notice: a requested item is ready. Log in through the normal library site if you need details.",
        ],
    },
    {
        "category": "legitimate_notice_calibration_v5",
        "risk": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "tactics": ["verification needed", "trusted route"],
        "impersonates": "known service",
        "pressure": "low-pressure routine notice",
        "ask": "check the normal account or known contact path",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Verify directly through the official app, official website, or a known phone number.",
        "messages": [
            "Your dentist appointment is confirmed for Tuesday at 2 PM. Reply C to confirm or call the office number you already have.",
            "Clinic reminder: please review your pre-visit forms in the patient portal before tomorrow.",
            "Auto shop update: your car is ready for pickup. Call our main shop number if you have questions.",
            "Apartment office: package pickup hours changed today. Check the resident portal for details.",
            "Library account: one item is due soon. Renew through the library website or front desk.",
            "School reminder: permission forms are due Friday. Check the parent portal for details.",
            "Transit card notice: your monthly pass renews tomorrow. Review it in your normal transit account.",
            "Gym notice: holiday hours changed. Open the member app if you want the schedule.",
            "Tax software: your draft return is ready for review. Sign in through the app you already use.",
            "Vet clinic: vaccine records are available in your pet portal.",
            "Bank statement ready: open your official banking app or website to view it.",
            "Pharmacy order shipped: track it through your normal pharmacy account.",
        ],
    },
    {
        "category": "benign_scary_calibration_v5",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": "known sender",
        "pressure": "no sensitive pressure",
        "ask": "no sensitive ask",
        "risk_text": "none identified",
        "safe_action": "If this came from someone you know and asks for nothing sensitive, it is probably safe. Still avoid unexpected links.",
        "messages": [
            "School alert: the lockdown drill is complete. Students are safe and classes have resumed.",
            "Clinic notice: lab results are available in the portal. No urgent action is required.",
            "Building notice: fire alarm testing starts at 3 PM. No evacuation is needed.",
            "Weather alert from school: after-school practice is cancelled. Pickup is at the usual door.",
            "Library reminder: your book hold expires Friday. No payment is due.",
            "Dentist reminder: appointment tomorrow morning. Call the office number on your calendar if needed.",
            "Office notice: security badges will be checked at the front desk tomorrow.",
            "Neighborhood update: roadwork begins Monday. Use the signed detour.",
            "Pharmacy reminder: refill is ready at your usual store.",
            "Court reminder: jury check-in starts Monday. Bring the summons you received in the mail.",
            "HR reminder: benefits enrollment closes Friday. Use the company portal.",
            "Utility notice: meter reading is scheduled this week. No appointment is required.",
        ],
    },
    {
        "category": "marketplace_calibration_v5",
        "risk": "suspicious",
        "scam_type": "marketplace_scam",
        "tactics": ["marketplace", "third party", "payment pressure"],
        "impersonates": "buyer or seller",
        "pressure": "unusual marketplace flow",
        "ask": "leave the marketplace, share codes, or handle third-party payment",
        "risk_text": "overpayment or account theft",
        "safe_action": "Pause before acting. Keep payment and messages inside the marketplace's official flow.",
        "messages": [
            "I can buy the couch today, but my cousin will pick it up and I will send extra money.",
            "The courier asks for a small insurance deposit. Pay it now and I will add it to your total.",
            "I sent a code to prove you are real. Send it back before I pay.",
            "My assistant pays by company email. Send your bank name so accounting can release funds.",
            "Please mark the item sold and move to text because my marketplace account is limited.",
            "Use this escrow page for the payment before pickup.",
            "I accidentally overpaid and need you to refund the difference to my driver.",
            "I am ready to buy but need your email to send a marketplace verification link.",
            "Can you hold the item if I send a check and arrange movers later?",
            "The buyer protection page says you need to pay a refundable seller fee.",
            "I am out of town, but I can send a courier if you pay their pickup fee first.",
            "Please verify the sale by reading me the account code you just received.",
        ],
    },
    {
        "category": "suspicious_urgency_calibration_v5",
        "risk": "suspicious",
        "scam_type": "unknown_urgent_request",
        "tactics": ["urgency", "vague request"],
        "impersonates": "unknown sender",
        "pressure": "act quickly without enough context",
        "ask": "respond, confirm, or review unclear issue",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Pause before responding. Verify through an official app, website, or known phone number.",
        "messages": [
            "Please call me back immediately about your account. This cannot wait.",
            "Final reminder: your paperwork is missing. Send your details by end of day.",
            "Action required: confirm ownership within 24 hours.",
            "We noticed an issue with your delivery. Respond quickly so we can resolve it.",
            "Your subscription has a problem. We need confirmation today or it may close.",
            "Urgent account message: please review the notice as soon as possible.",
            "We need a response before 5 PM to avoid delays.",
            "Your request cannot be completed until you confirm a few details.",
            "Important: your file is on hold. Reply when available.",
            "Your account review is pending. A response is needed today.",
        ],
    },
    {
        "category": "dangerous_contrast_calibration_v5",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["fake authority", "credential request", "suspicious link", "urgency"],
        "impersonates": "service provider",
        "pressure": "account, payment, or delivery emergency",
        "ask": "use a message link, message number, code, card, or login",
        "risk_text": "credential or payment theft",
        "safe_action": "Do not use links or phone numbers from the message. Open the official app or website yourself.",
        "messages": [
            "Bank alert: new device signed in. Verify your password at http://bank-security-review.example within 10 minutes.",
            "Credit union: card locked. Call the number in this text and read your one-time code to cancel fraud.",
            "Delivery center: package held for a $2.19 fee. Enter address and card at http://parcel-release-fee.example.",
            "PayPal warning: passkey added. Secure your account at https://paypal-case-review.example and enter the code.",
            "Hospital billing: results blocked until you upload ID and insurance number at this link.",
            "Utility shutoff: pay through this urgent portal today or service will end.",
            "County tax final notice: pay the penalty fee through the link below before noon.",
            "Airline refund desk: provide card details at this refund link to keep your ticket.",
            "School portal emergency form missing. Enter parent login and payment card at this link.",
            "Streaming billing: payment failed and your account closes tonight. Update at http://streaming-billing-review.example.",
            "Card fraud center: suspicious grocery purchase. Call this alert number and verify your full card number.",
            "Coinbase security: phone update requested. Contact support at the callback number in this text.",
        ],
    },
]


def calibration_cases(repeats: int) -> list[tuple[str, dict, dict]]:
    cases = []
    for repeat in range(repeats):
        for case in CALIBRATION_CASES:
            prediction = hard_prediction(case)
            for index, message in enumerate(case["messages"]):
                case_id = f"{case['category']}_{repeat:02d}_{index:02d}"
                cases.append((case_id, {"message": message, "scenario": case}, prediction))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jawbreaker v5 1B calibration data.")
    parser.add_argument("--base-train", type=int, default=680)
    parser.add_argument("--base-dev", type=int, default=120)
    parser.add_argument("--base-test", type=int, default=180)
    parser.add_argument("--hard-repeats", type=int, default=4)
    parser.add_argument("--boundary-repeats", type=int, default=5)
    parser.add_argument("--v4-repeats", type=int, default=5)
    parser.add_argument("--calibration-repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "eval" / "hard_v5_eval.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.base_train + args.base_dev + args.base_test
    base = list(generated_cases(total))
    v2_hard = hard_cases(args.hard_repeats)
    boundary = boundary_cases(args.boundary_repeats)
    v4 = v4_cases(args.v4_repeats)
    calibration = calibration_cases(args.calibration_repeats)

    v2_hard_train, v2_hard_dev = split_special_cases(v2_hard, dev_fraction=0.20)
    boundary_train, boundary_dev = split_special_cases(boundary, dev_fraction=0.20)
    v4_train, v4_dev = split_special_cases(v4, dev_fraction=0.20)
    calibration_train, calibration_dev = split_special_cases(calibration, dev_fraction=0.20)

    train_cases = base[: args.base_train] + v2_hard_train + boundary_train + v4_train + calibration_train
    dev_cases = (
        base[args.base_train : args.base_train + args.base_dev]
        + v2_hard_dev
        + boundary_dev
        + v4_dev
        + calibration_dev
    )
    test_cases = (
        base[args.base_train + args.base_dev :]
        + v2_hard_dev
        + boundary_dev
        + v4_dev
        + calibration_dev
    )

    write_jsonl(args.out_dir / "train_v5.jsonl", (sft_row(*case) for case in train_cases))
    write_jsonl(args.out_dir / "dev_v5.jsonl", (sft_row(*case) for case in dev_cases))
    write_jsonl(args.out_dir / "test_v5.jsonl", (sft_row(*case) for case in test_cases))
    write_jsonl(args.eval_out, (eval_row(*case) for case in test_cases))

    print(f"wrote train_v5={len(train_cases)} dev_v5={len(dev_cases)} test_v5={len(test_cases)}")
    print(f"v2_hard={len(v2_hard)} boundary={len(boundary)} v4={len(v4)} calibration={len(calibration)}")
    print(f"calibration_train={len(calibration_train)} calibration_dev={len(calibration_dev)}")
    print(f"training data: {args.out_dir}")
    print(f"hard eval: {args.eval_out}")


if __name__ == "__main__":
    main()
