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
from training.generate_v5_data import calibration_cases as v5_calibration_cases  # noqa: E402


V6_CALIBRATION_CASES = [
    {
        "category": "safe_scary_no_action_v6",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": "known sender",
        "pressure": "no sensitive pressure",
        "ask": "no sensitive ask",
        "risk_text": "none identified",
        "safe_action": (
            "If this came from someone you know and asks for nothing sensitive, it is probably safe. "
            "Still avoid unexpected links."
        ),
        "messages": [
            "Clinic notice: lab results were posted to the usual patient portal. No urgent action is required.",
            "School safety update: the drill is complete, students are safe, and the normal schedule has resumed.",
            "Building notice: fire alarm testing is scheduled today. No evacuation or reply is needed.",
            "Weather update: buses are delayed by fifteen minutes. Pickup is still at the normal stop.",
            "Library notice: your requested book is ready at the front desk. No fee is due.",
            "HR reminder: open enrollment closes Friday. Use the company portal if you want to make changes.",
            "Court reminder: jury check-in begins Monday. Bring the summons you received in the mail.",
            "Pharmacy update: your refill is ready at your usual store.",
            "Dentist reminder: your appointment is tomorrow. Call the office number from your calendar if needed.",
            "Neighborhood alert: roadwork starts Monday. Follow the posted detour signs.",
            "Utility notice: meter reading happens this week. No appointment or payment is required.",
            "Office security update: badge checks will happen at the front desk tomorrow.",
            "School alert: soccer practice is cancelled because of weather. Students should go home as usual.",
            "Clinic reminder: fasting is recommended before tomorrow's appointment. Use your normal portal for forms.",
            "Library reminder: one book is due soon. Return it at the desk or renew through the usual account.",
            "Apartment notice: package room hours changed this week. Ask the front desk if you need help.",
        ],
    },
    {
        "category": "needs_check_official_route_v6",
        "risk": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "tactics": ["verification needed", "trusted route"],
        "impersonates": "known service",
        "pressure": "routine account or safety notice",
        "ask": "verify through official app, official website, or known route",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Verify directly through the official app, official website, or a known phone number.",
        "messages": [
            "Bank security: a new device signed in. Open the official banking app yourself to review activity.",
            "Credit union notice: contact information changed. Use the number on your card if this was not you.",
            "Card fraud alert: did you attempt a $68.45 grocery purchase? Reply YES or NO, or call the number on your card.",
            "Coinbase notice: your password was changed. Open the official app directly to check security.",
            "PayPal notice: a new passkey was added. Visit paypal.com directly if this was not you.",
            "Streaming service: your payment method needs attention. Open the official app or website yourself.",
            "Delivery carrier: weather delayed your package. Track it only in the official carrier app.",
            "Utility reminder: your statement is ready. Pay through your normal account or printed bill number.",
            "Insurance portal: an explanation of benefits is ready. Review it in your usual member portal.",
            "County tax office: your annual notice is available. Use the official county website from prior mail.",
            "Airline notice: your flight time changed. Open the airline app directly to confirm.",
            "Hospital portal: new results are available. Sign in through your usual patient portal.",
            "School portal: report cards are available. Use the normal parent portal.",
            "Phone carrier: SIM protection changed. Open the carrier app directly to review account settings.",
            "Pharmacy account: order details were updated. Check through your normal pharmacy login.",
            "Library account: a hold expires soon. Renew through the library website or front desk.",
        ],
    },
    {
        "category": "suspicious_not_dangerous_marketplace_v6",
        "risk": "suspicious",
        "scam_type": "marketplace_scam",
        "tactics": ["marketplace", "third party", "payment pressure"],
        "impersonates": "buyer or seller",
        "pressure": "unusual marketplace flow",
        "ask": "leave the marketplace flow, use a third party, or handle unclear payment",
        "risk_text": "overpayment or account theft",
        "safe_action": "Pause before acting. Keep payment and messages inside the marketplace's official flow.",
        "messages": [
            "I want the table but my cousin will pick it up. I can send extra money now.",
            "Can you hold the chair if I mail a check and arrange movers later?",
            "My assistant handles payment by email. Send your bank name so accounting can prepare it.",
            "I am ready to buy but need your email to send the marketplace confirmation.",
            "Please move to text because my marketplace account has a limit today.",
            "The courier needs a refundable pickup fee. I can include it in the final amount.",
            "I accidentally sent too much. Can you return the difference to the driver?",
            "The buyer protection page says there is a refundable seller verification step.",
            "I am out of town, but a delivery company can collect it if you coordinate with them.",
            "Can you mark it sold before payment? My pickup person is nearby.",
            "I use a third-party escrow service for all marketplace purchases.",
            "Please confirm your email and phone so my payment department can send instructions.",
            "I sent an account code to prove the listing is real. Please read it back before I pay.",
            "I can pay today if you leave the marketplace chat and use this payment flow.",
            "The shipping agent will contact you about a small insurance hold.",
            "My company buyer will send a certified check and arrange pickup after it clears.",
        ],
    },
    {
        "category": "suspicious_not_dangerous_link_v6",
        "risk": "suspicious",
        "scam_type": "unknown_link",
        "tactics": ["suspicious link", "vague request"],
        "impersonates": "unknown sender",
        "pressure": "unclear account or document issue",
        "ask": "open a link or review unclear issue",
        "risk_text": "possible credential theft",
        "safe_action": (
            "Do not use the link from the message. Open the official website or app yourself if you need to check."
        ),
        "messages": [
            "Your profile needs attention. Review details at http://profile-message-center.example.",
            "We could not process your request. Open https://customer-resolution.example for more information.",
            "A secure document is waiting for review at https://document-viewer-secure.example.",
            "Your voicemail is available at http://voice-message-download.example.",
            "Your account notice is ready. Visit http://account-message-center.example when you have time.",
            "Please confirm your information through http://profile-update-center.example.",
            "A shared file requires review: https://document-review-center.example.",
            "Your delivery update is available here: http://package-info-review.example.",
            "Your subscription message can be viewed at https://member-notice-center.example.",
            "There is a pending form in your inbox: http://form-review-portal.example.",
            "A billing message is available at https://billing-message-view.example.",
            "Your support case was updated. See http://support-case-message.example.",
        ],
    },
    {
        "category": "dangerous_anchor_v6",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["fake authority", "credential request", "suspicious link", "urgency"],
        "impersonates": "service provider",
        "pressure": "account, payment, or delivery emergency",
        "ask": "use a message link, message number, code, card, or login",
        "risk_text": "credential or payment theft",
        "safe_action": "Do not use links or phone numbers from the message. Open the official app or website yourself.",
        "messages": [
            "Bank security: new device signed in. Enter your password at http://bank-device-review.example now.",
            "Credit union: card locked. Call the number in this text and read your one-time code.",
            "Card fraud center: suspicious purchase. Verify your full card number through this alert number.",
            "Delivery center: package held for a $2.19 fee. Enter address and card at http://parcel-fee-release.example.",
            "PayPal security: passkey added. Secure your account at https://paypal-case-review.example and enter the code.",
            "Coinbase support: phone update requested. Contact the callback number in this text immediately.",
            "Streaming billing: payment failed and your account closes tonight. Update at http://streaming-billing-review.example.",
            "Utility shutoff warning: pay through this urgent portal today or service will end.",
            "Hospital billing: results blocked until you upload ID and insurance number at this link.",
            "School portal emergency form missing. Enter parent login and payment card at this link.",
            "Airline refund desk: provide card details at this refund link to keep your ticket.",
            "County tax final notice: pay the penalty fee through the link below before noon.",
            "Remote support alert: call the number shown and read the desktop access code to fix your device.",
            "Device protection: install the remote cleanup tool and pay the service fee before data loss.",
            "Grandma, this is my new number. Send money today and please do not tell anyone.",
            "TikTok Shop recruiter: earn $720 daily. Message us on WhatsApp and pay the training deposit.",
        ],
    },
]


def v6_calibration_cases(repeats: int) -> list[tuple[str, dict, dict]]:
    cases = []
    for repeat in range(repeats):
        for case in V6_CALIBRATION_CASES:
            prediction = hard_prediction(case)
            for index, message in enumerate(case["messages"]):
                case_id = f"{case['category']}_{repeat:02d}_{index:02d}"
                cases.append((case_id, {"message": message, "scenario": case}, prediction))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jawbreaker v6 final 1B calibration data.")
    parser.add_argument("--base-train", type=int, default=680)
    parser.add_argument("--base-dev", type=int, default=120)
    parser.add_argument("--base-test", type=int, default=180)
    parser.add_argument("--hard-repeats", type=int, default=4)
    parser.add_argument("--boundary-repeats", type=int, default=5)
    parser.add_argument("--v4-repeats", type=int, default=5)
    parser.add_argument("--v5-repeats", type=int, default=5)
    parser.add_argument("--v6-repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "eval" / "hard_v6_eval.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.base_train + args.base_dev + args.base_test
    base = list(generated_cases(total))
    v2_hard = hard_cases(args.hard_repeats)
    boundary = boundary_cases(args.boundary_repeats)
    v4 = v4_cases(args.v4_repeats)
    v5 = v5_calibration_cases(args.v5_repeats)
    v6 = v6_calibration_cases(args.v6_repeats)

    v2_hard_train, v2_hard_dev = split_special_cases(v2_hard, dev_fraction=0.20)
    boundary_train, boundary_dev = split_special_cases(boundary, dev_fraction=0.20)
    v4_train, v4_dev = split_special_cases(v4, dev_fraction=0.20)
    v5_train, v5_dev = split_special_cases(v5, dev_fraction=0.20)
    v6_train, v6_dev = split_special_cases(v6, dev_fraction=0.20)

    train_cases = base[: args.base_train] + v2_hard_train + boundary_train + v4_train + v5_train + v6_train
    dev_cases = (
        base[args.base_train : args.base_train + args.base_dev]
        + v2_hard_dev
        + boundary_dev
        + v4_dev
        + v5_dev
        + v6_dev
    )
    test_cases = (
        base[args.base_train + args.base_dev :]
        + v2_hard_dev
        + boundary_dev
        + v4_dev
        + v5_dev
        + v6_dev
    )

    write_jsonl(args.out_dir / "train_v6.jsonl", (sft_row(*case) for case in train_cases))
    write_jsonl(args.out_dir / "dev_v6.jsonl", (sft_row(*case) for case in dev_cases))
    write_jsonl(args.out_dir / "test_v6.jsonl", (sft_row(*case) for case in test_cases))
    write_jsonl(args.eval_out, (eval_row(*case) for case in test_cases))

    print(f"wrote train_v6={len(train_cases)} dev_v6={len(dev_cases)} test_v6={len(test_cases)}")
    print(f"v2_hard={len(v2_hard)} boundary={len(boundary)} v4={len(v4)} v5={len(v5)} v6={len(v6)}")
    print(f"v6_train={len(v6_train)} v6_dev={len(v6_dev)}")
    print(f"training data: {args.out_dir}")
    print(f"hard eval: {args.eval_out}")


if __name__ == "__main__":
    main()
