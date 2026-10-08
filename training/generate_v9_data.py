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
from training.generate_v6_data import v6_calibration_cases  # noqa: E402
from training.generate_v7_data import v7_calibration_cases  # noqa: E402
from training.generate_v8_data import v8_calibration_cases  # noqa: E402


V9_CALIBRATION_CASES = [
    {
        "category": "zero_width_evasion_danger_v9",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["zero_width_evasion", "credential request", "fake authority"],
        "impersonates": "bank or online platform",
        "pressure": "invisible unicode zero-width characters disguise phishing keywords",
        "ask": "log in or verify credentials on an obfuscated link",
        "risk_text": "credential theft or account compromise",
        "safe_action": "Do not click links or enter passwords. Contact support using an official app or known website.",
        "messages": [
            "Us\u200bps account notification: U\u200brgent password ve\u200brification required at http://usps-track-secure.example",
            "C\u200bha\u200bse Security Notice: Un\u200bauthorized login attempt detected. Confirm your identity at https://paypa1.com",
            "G\u200bma\u200bil alert: Your account will be suspended in 24 hours. Verify credentials at http://192.168.1.1/login",
            "Am\u200bazon alert: Unrecognized sign-in detected. Enter your code at https://amazon.com@evil-phish.net/login",
            "N\u200betflix billing issue: Your payment card was declined. Update card details at https://bit.ly/3xYz90",
        ],
    },
    {
        "category": "mfa_fatigue_callback_v9",
        "risk": "dangerous",
        "scam_type": "callback_phishing",
        "tactics": ["fake authority", "callback request", "account takeover"],
        "impersonates": "bank or security department",
        "pressure": "fake password reset alert encouraging victim to call a fraud number",
        "ask": "call the fraudulent callback number or read the one-time passcode",
        "risk_text": "account takeover or credential loss",
        "safe_action": "Do not call numbers from unverified text messages. Call the number on the back of your card.",
        "messages": [
            "Coinbase Security: Did you request a phone number change? If not, call fraud prevention immediately at 800-555-0199.",
            "Bank of America Alert: A wire transfer of $1,450 to Zelle is pending. If you did not authorize this, call 888-555-0122.",
            "Wells Fargo Notice: Unauthorized password reset initiated. Reply to approve or call support at 800-555-0144.",
            "Microsoft Defender: Threat detected on your online account. Call the listed support desk number immediately.",
            "PayPal Security: Did you attempt a login from Moscow? If this was not you, call 800-555-0188 now.",
        ],
    },
    {
        "category": "toll_delivery_fee_smishing_v9",
        "risk": "dangerous",
        "scam_type": "payment_request",
        "tactics": ["fake authority", "payment pressure", "suspicious link"],
        "impersonates": "toll road authority or postal carrier",
        "pressure": "fake unpaid toll invoice or package delivery fee with late penalty threats",
        "ask": "pay an unpaid fee or enter card details via a web link",
        "risk_text": "credit card theft or financial loss",
        "safe_action": "Do not pay fees via SMS links. Visit the official toll or carrier website directly.",
        "messages": [
            "SunPass Alert: You have an unpaid toll invoice of $4.85. Pay before midnight to avoid a $50 fine: https://bit.ly/3xYz90",
            "EZPass Notice: Final warning for unpaid toll fees of $3.50. Settle your account now at http://ezpass-toll-pay.example",
            "USPS Delivery: Package held due to an unpaid redelivery fee of $1.99. Pay here: http://usps-redelivery.example",
            "FedEx Shipping: Parcel delivery on hold due to missing customs tax. Pay fee: http://fedex-customs.example",
            "Toll Services: Unpaid toll road bill of $6.20. Settle invoice immediately at http://104.28.1.1/pay",
        ],
    },
    {
        "category": "quishing_qr_code_v9",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["quishing_qr_prompt", "suspicious link", "fake authority"],
        "impersonates": "bank, HR department, or service provider",
        "pressure": "fake urgency to scan a QR code for security or payment verification",
        "ask": "scan the QR code using phone camera and enter credentials",
        "risk_text": "credential theft or malware redirection",
        "safe_action": "Do not scan unknown QR codes sent via text or email. Log in directly on the official app.",
        "messages": [
            "IT Security Alert: Scan the QR code below to verify your 2FA token before your corporate email is locked.",
            "Parking Services: Scan QR code to pay your outstanding parking fee online immediately.",
            "HR Notice: Annual payroll setup requires you to scan the QR code and confirm banking details.",
            "Bank Security: Scan the QR code to re-authenticate your mobile banking app.",
            "Rewards Dept: You won a $100 gift voucher! Scan the QR code to claim your reward.",
        ],
    },
]


def v9_calibration_cases(repeats: int) -> list[tuple[str, dict, dict]]:
    cases = []
    for repeat in range(repeats):
        for case in V9_CALIBRATION_CASES:
            prediction = hard_prediction(case)
            for index, message in enumerate(case["messages"]):
                case_id = f"{case['category']}_{repeat:02d}_{index:02d}"
                cases.append((case_id, {"message": message, "scenario": case}, prediction))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jawbreaker v9 hard case calibration data.")
    parser.add_argument("--base-train", type=int, default=640)
    parser.add_argument("--base-dev", type=int, default=110)
    parser.add_argument("--base-test", type=int, default=170)
    parser.add_argument("--hard-repeats", type=int, default=4)
    parser.add_argument("--boundary-repeats", type=int, default=5)
    parser.add_argument("--v4-repeats", type=int, default=5)
    parser.add_argument("--v5-repeats", type=int, default=5)
    parser.add_argument("--v6-repeats", type=int, default=5)
    parser.add_argument("--v7-repeats", type=int, default=5)
    parser.add_argument("--v8-repeats", type=int, default=5)
    parser.add_argument("--v9-repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "eval" / "hard_v9_eval.jsonl")
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
    v7 = v7_calibration_cases(args.v7_repeats)
    v8 = v8_calibration_cases(args.v8_repeats)
    v9 = v9_calibration_cases(args.v9_repeats)

    v2_hard_train, v2_hard_dev = split_special_cases(v2_hard, dev_fraction=0.20)
    boundary_train, boundary_dev = split_special_cases(boundary, dev_fraction=0.20)
    v4_train, v4_dev = split_special_cases(v4, dev_fraction=0.20)
    v5_train, v5_dev = split_special_cases(v5, dev_fraction=0.20)
    v6_train, v6_dev = split_special_cases(v6, dev_fraction=0.20)
    v7_train, v7_dev = split_special_cases(v7, dev_fraction=0.20)
    v8_train, v8_dev = split_special_cases(v8, dev_fraction=0.20)
    v9_train, v9_dev = split_special_cases(v9, dev_fraction=0.20)

    train_cases = (
        base[: args.base_train]
        + v2_hard_train
        + boundary_train
        + v4_train
        + v5_train
        + v6_train
        + v7_train
        + v8_train
        + v9_train
    )
    dev_cases = (
        base[args.base_train : args.base_train + args.base_dev]
        + v2_hard_dev
        + boundary_dev
        + v4_dev
        + v5_dev
        + v6_dev
        + v7_dev
        + v8_dev
        + v9_dev
    )
    test_cases = (
        base[args.base_train + args.base_dev :]
        + v2_hard_dev
        + boundary_dev
        + v4_dev
        + v5_dev
        + v6_dev
        + v7_dev
        + v8_dev
        + v9_dev
    )

    write_jsonl(args.out_dir / "train_v9.jsonl", (sft_row(*case) for case in train_cases))
    write_jsonl(args.out_dir / "dev_v9.jsonl", (sft_row(*case) for case in dev_cases))
    write_jsonl(args.out_dir / "test_v9.jsonl", (sft_row(*case) for case in test_cases))
    write_jsonl(args.eval_out, (eval_row(*case) for case in test_cases))

    print(f"wrote train_v9={len(train_cases)} dev_v9={len(dev_cases)} test_v9={len(test_cases)}")
    print(f"v9 hard eval dataset generated at: {args.eval_out}")


if __name__ == "__main__":
    main()
