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


BOUNDARY_CASES = [
    {
        "category": "bank_credential_boundary",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["fake authority", "credential request", "urgency"],
        "impersonates": "bank security",
        "pressure": "account or card access threatened",
        "ask": "share a PIN, one-time code, card number, or password",
        "risk_text": "credential theft",
        "safe_action": "Do not reply with codes, passwords, PINs, or card numbers. Open the official banking app or call the number on your card.",
        "messages": [
            "Card Security: your debit card is suspended. Confirm your PIN now to restore access.",
            "Chase verification: unusual transfer detected. Reply with your one-time code to stop it.",
            "Credit union alert: your card will close today unless you send your full card number.",
            "Bank of America security: profile expires today. Send password to keep account active.",
            "Fraud desk: we received a request to change your account phone number. Call the number in this text now.",
            "Coinbase security: phone number update requested. If this was not you, call support at the number below.",
            "Zelle support: transfer pending. Reply with the verification code so we can cancel it.",
            "Card services: suspicious login found. Confirm your billing information in this message.",
        ],
    },
    {
        "category": "package_fee_boundary",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["fake authority", "payment request", "suspicious link", "urgency"],
        "impersonates": "delivery carrier",
        "pressure": "delivery held or final attempt",
        "ask": "pay a fee or verify personal details through a message link",
        "risk_text": "credential or payment theft",
        "safe_action": "Do not click the delivery link or pay through the message. Open the carrier website yourself or call a number you already trust.",
        "messages": [
            "Postal notice: your package is held for an unpaid customs fee. Pay now at http://postal-release.example",
            "DHL: final delivery attempt failed. Enter address and card details to reschedule.",
            "USPS: incomplete address. Verify your identity and pay $1.99 to release the parcel.",
            "UPS support: package will return to sender unless you confirm billing info today.",
            "FedEx: delivery tax unpaid. Use this secure link to avoid storage charges.",
            "Package Center: your item is waiting. Enter birth date and payment card to continue.",
            "USPS track: unpaid postage fee. Verify now using the link in this text.",
            "Courier desk: insurance payment required before driver pickup. We will refund it later.",
        ],
    },
    {
        "category": "tech_support_boundary",
        "risk": "dangerous",
        "scam_type": "tech_support",
        "tactics": ["fake authority", "remote access request", "payment request", "urgency"],
        "impersonates": "tech support",
        "pressure": "device or account emergency",
        "ask": "install remote access, share a code, call the message number, or pay a fee",
        "risk_text": "remote access or payment theft",
        "safe_action": "Do not call the number or install remote access tools. Close the message and contact official support through a trusted website.",
        "messages": [
            "Microsoft case opened: malware found. Call this technician now and share the screen code.",
            "Apple support: your iCloud is locked. Reply with the six digit code to verify ownership.",
            "Security alert: antivirus cleanup fee is overdue. Pay now to prevent shutdown.",
            "Windows Defender: your computer is compromised. Install this support tool immediately.",
            "Geek support: refund pending. Open the remote support link so we can return your money.",
            "Device protection team: subscription expired. Call the number in this alert to avoid data loss.",
            "Support desk: we detected a virus. Technician requires gift cards to unlock protection.",
            "Account security: share the login code so we can remove the attacker.",
        ],
    },
    {
        "category": "prize_fee_boundary",
        "risk": "dangerous",
        "scam_type": "prize_scam",
        "tactics": ["too good to be true", "payment request", "credential request"],
        "impersonates": "reward or grant office",
        "pressure": "reward expires or prize is waiting",
        "ask": "pay a fee or share banking details",
        "risk_text": "money or credential theft",
        "safe_action": "Do not pay fees or share financial details to claim a prize. Verify through an official source you find yourself.",
        "messages": [
            "Grant office: your $8,500 grant is approved. Pay the processing fee to release funds.",
            "Reward Center: unclaimed prize expires tonight. Send your bank login for direct deposit.",
            "Sweepstakes team: winner selected. Buy an activation card before 6 PM to claim.",
            "Lottery desk: congratulations. Pay taxes with gift cards before your prize can be delivered.",
            "Travel rewards: free vacation approved. Enter your card number to reserve the certificate.",
            "Cash award notice: release requires a small wire fee today.",
            "Government benefits team: emergency grant waiting. Send account details for verification.",
            "Prize support: claim at this link and pay shipping before midnight.",
        ],
    },
    {
        "category": "job_scam_boundary",
        "risk": "dangerous",
        "scam_type": "job_scam",
        "tactics": ["fake job", "payment request", "too good to be true", "suspicious contact"],
        "impersonates": "recruiter",
        "pressure": "fast hiring or high pay",
        "ask": "pay fees, buy equipment, move money, or contact through unofficial chat",
        "risk_text": "advance fee or check scam",
        "safe_action": "Do not pay to start a job or move money for an employer. Verify the company through its official website.",
        "messages": [
            "TikTok Shop recruiter: part-time assistant pays $650 per day. Contact us on WhatsApp to begin.",
            "Remote job approved. Deposit our check, buy equipment, and send the remaining money back.",
            "Payroll team: your role starts today. Pay an onboarding fee so we can activate your account.",
            "Hiring manager: we only interview on Telegram. Send your bank login for direct payroll setup.",
            "Online task work: earn $720 daily for one hour of work. Message this number to receive tasks.",
            "Recruiting desk: equipment shipment is ready. Buy from this link and we reimburse later.",
            "Work from home coordinator: complete training fee payment before your first assignment.",
            "HR assistant: receive customer funds in your account and forward them to our vendor.",
        ],
    },
    {
        "category": "family_romance_boundary",
        "risk": "dangerous",
        "scam_type": "family_impersonation",
        "tactics": ["impersonation", "secrecy", "payment pressure", "urgency"],
        "impersonates": "family member or romantic interest",
        "pressure": "emergency and secrecy",
        "ask": "send money, crypto, wire transfer, or gift cards",
        "risk_text": "money theft",
        "safe_action": "Do not send money. Call the person using a number you already know or ask another trusted person to verify.",
        "messages": [
            "Auntie, my phone broke. Use this number only and wire emergency funds before Dad finds out.",
            "Grandma it is me from a temporary phone. I need crypto for rent today and cannot talk.",
            "Do not tell anyone about us yet. I need help with a hospital bill before we can meet.",
            "Mom, please keep this quiet. Buy Apple gift cards and text me the codes.",
            "This is your nephew. I am stranded and need a payment app transfer right away.",
            "I love you but my account is frozen. Send gift cards today and I will explain tomorrow.",
            "Please do not call my old number. I need wire money for bail now.",
            "Your daughter asked me to text you. Send funds to this wallet for her emergency.",
        ],
    },
    {
        "category": "official_route_contrast",
        "risk": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "tactics": ["verification needed", "trusted route"],
        "impersonates": "service provider",
        "pressure": "routine account notice",
        "ask": "verify through official app or known route",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Verify directly through the official app, official website, or a known phone number.",
        "messages": [
            "Your bank statement is ready. Open the official banking app to view it.",
            "Fraud alert: did you attempt a $249.00 purchase at Target? Reply YES or NO.",
            "Your Coinbase password was changed. If this was not you, open the official app to review security.",
            "UPS update: your package is out for delivery. Track it in the UPS app.",
            "Your pharmacy order shipped. Open your normal pharmacy account for details.",
            "Your utility bill is ready. No payment is due until next week through your normal account.",
            "School reminder: permission forms are due Friday. Check the parent portal.",
            "Your verification code is 482910. Do not share this code with anyone.",
        ],
    },
    {
        "category": "benign_scary_contrast",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": "known sender",
        "pressure": "no pressure",
        "ask": "no sensitive ask",
        "risk_text": "none identified",
        "safe_action": "If this came from someone you know and asks for nothing sensitive, it is probably safe. Still avoid unexpected links.",
        "messages": [
            "Dentist reminder: appointment tomorrow at 2 PM. Call the office number you already have if needed.",
            "Can you bring milk on your way home? No rush.",
            "Your library book is ready for pickup at the front desk.",
            "Dinner moved to 6 PM. No need to bring anything.",
            "Your clinic lab results are available in the patient portal.",
            "Thanks for helping today. I left your receipt on the kitchen table.",
            "Soccer practice is cancelled because of weather.",
            "The school bus is running ten minutes late this morning.",
        ],
    },
]


def boundary_cases(repeats: int) -> list[tuple[str, dict, dict]]:
    cases = []
    for repeat in range(repeats):
        for case in BOUNDARY_CASES:
            prediction = hard_prediction(case)
            for index, message in enumerate(case["messages"]):
                case_id = f"{case['category']}_{repeat:02d}_{index:02d}"
                cases.append((case_id, {"message": message, "scenario": case}, prediction))
    return cases


def split_special_cases(cases: list[tuple[str, dict, dict]], dev_fraction: float) -> tuple[list, list]:
    split = max(1, round(len(cases) * dev_fraction))
    return cases[split:], cases[:split]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jawbreaker v3 contrastive boundary training data.")
    parser.add_argument("--base-train", type=int, default=680)
    parser.add_argument("--base-dev", type=int, default=120)
    parser.add_argument("--base-test", type=int, default=180)
    parser.add_argument("--hard-repeats", type=int, default=4)
    parser.add_argument("--boundary-repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "eval" / "hard_v3_eval.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.base_train + args.base_dev + args.base_test
    base = list(generated_cases(total))
    v2_hard = hard_cases(args.hard_repeats)
    boundary = boundary_cases(args.boundary_repeats)

    v2_hard_train, v2_hard_dev = split_special_cases(v2_hard, dev_fraction=0.20)
    boundary_train, boundary_dev = split_special_cases(boundary, dev_fraction=0.20)

    train_cases = base[: args.base_train] + v2_hard_train + boundary_train
    dev_cases = base[args.base_train : args.base_train + args.base_dev] + v2_hard_dev + boundary_dev
    test_cases = base[args.base_train + args.base_dev :] + v2_hard_dev + boundary_dev

    write_jsonl(args.out_dir / "train_v3.jsonl", (sft_row(*case) for case in train_cases))
    write_jsonl(args.out_dir / "dev_v3.jsonl", (sft_row(*case) for case in dev_cases))
    write_jsonl(args.out_dir / "test_v3.jsonl", (sft_row(*case) for case in test_cases))
    write_jsonl(args.eval_out, (eval_row(*case) for case in test_cases))

    print(f"wrote train_v3={len(train_cases)} dev_v3={len(dev_cases)} test_v3={len(test_cases)}")
    print(f"v2_hard={len(v2_hard)} v2_hard_train={len(v2_hard_train)} v2_hard_dev={len(v2_hard_dev)}")
    print(f"boundary={len(boundary)} boundary_train={len(boundary_train)} boundary_dev={len(boundary_dev)}")
    print(f"training data: {args.out_dir}")
    print(f"hard eval: {args.eval_out}")


if __name__ == "__main__":
    main()
