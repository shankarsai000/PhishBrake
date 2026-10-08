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


V4_CASES = [
    {
        "category": "tech_support_long_json_lock",
        "risk": "dangerous",
        "scam_type": "tech_support",
        "tactics": ["fake authority", "remote access request", "urgency"],
        "impersonates": "tech support",
        "pressure": "device or account emergency",
        "ask": "call the message number, share a code, or install remote tools",
        "risk_text": "remote access or credential theft",
        "safe_action": "Do not call the number or install remote access tools. Contact official support through a trusted website.",
        "messages": [
            "Windows Security Center: multiple banking trojans were detected during a routine scan. Your device protection subscription is expired and your files may be deleted. Do not turn off the computer. Call the support number in this alert and give the technician the screen code so they can remove the infection.",
            "Apple Device Care: suspicious sign-in activity was found and your iCloud backup may be disabled. A technician is waiting to verify ownership. Call the number shown, keep this case open, and read the six digit code before data loss occurs.",
            "Browser warning: your computer has been blocked for safety reasons. The page says not to restart, not to close the tab, and to call certified support immediately. The support agent asks you to install a remote cleanup tool and pay the unlock fee.",
            "Geek support renewal: your service plan was charged by mistake. To refund it, open the remote support link, allow the technician to view your screen, and confirm the bank account where the refund should go.",
            "Antivirus response team: malware is sending photos and saved passwords from this device. Call the emergency number in the pop-up, share the desktop access code, and pay the cleanup deposit with a prepaid card.",
            "Account recovery desk: your email was used from another country. To stop permanent closure, call this callback number, verify the code sent to your phone, and keep the phone line open during repair.",
            "Microsoft defender audit: unusual command activity found. A technician needs remote access to isolate the device. The alert says the case will be closed in 20 minutes unless you call the listed support number.",
            "Security billing office: your protection plan is expired and the renewal failed. The message asks you to download a support app, enter your card number, and read the confirmation code to the operator.",
            "Cloud storage support: backup corruption detected. To avoid losing family photos, call the number in this notice and provide the account reset code that arrives by text.",
            "Device warranty alert: unauthorized login found. The operator says they can fix it only if you install remote access software and leave the computer connected while they process a service fee.",
            "Printer support case: driver error locked your network. Call the help desk number in this message, grant screen sharing, and buy a troubleshooting voucher before the case expires.",
            "Streaming device support: subscription expired and a security issue was found. The message asks you to call the number shown and provide the one-time code to reactivate protection.",
        ],
    },
    {
        "category": "official_route_v4",
        "risk": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "tactics": ["verification needed", "trusted route"],
        "impersonates": "service provider",
        "pressure": "routine account or safety notice",
        "ask": "verify through official app, official website, or known route",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Verify directly through the official app, official website, or a known phone number.",
        "messages": [
            "Netflix notice: your payment method needs attention. Open the Netflix app or website yourself to update billing.",
            "Credit union alert: we noticed a new login. If this was not you, open the official mobile app and review security.",
            "Hospital portal: new test results are available. Sign in through your usual patient portal to view them.",
            "Utility reminder: your statement is ready. Pay only through your normal account or the phone number printed on your bill.",
            "Airline update: your flight time changed. Open the airline app or website directly to review the itinerary.",
            "County tax office: your annual notice is available. Visit the official county website printed on prior mail.",
            "Delivery update: your package is delayed because of weather. Check tracking in the official carrier app.",
            "Card fraud alert: did you attempt a $318.44 grocery purchase? Reply YES or NO, or call the number on your card.",
            "Insurance EOB ready: this is not a bill. Review details in your normal insurance portal.",
            "School safety notice: pickup procedures changed today. Check the parent portal for official instructions.",
        ],
    },
    {
        "category": "external_route_phishing_v4",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["fake authority", "suspicious link", "urgency"],
        "impersonates": "service provider",
        "pressure": "account or payment emergency",
        "ask": "use the message link or phone number",
        "risk_text": "credential or payment theft",
        "safe_action": "Do not use links or phone numbers from the message. Open the official app or website yourself.",
        "messages": [
            "Netflix billing: payment failed and your account closes tonight. Update now at http://netflix-billing-review.example",
            "Credit union security: new login detected. Call the fraud number in this text now to cancel it.",
            "Hospital portal: test results blocked. Enter your login and insurance number here: https://patient-records-release.example",
            "Utility shutoff warning: service stops today unless you pay through this urgent portal: http://utility-fastpay.example",
            "Airline refund desk: your ticket changed. Provide card details at https://airline-refund-claim.example",
            "County tax office: final notice. Pay the penalty fee with the link below before noon.",
            "Carrier delivery: weather delay fee unpaid. Confirm address and card at http://package-release-fee.example",
            "Card fraud center: suspicious grocery purchase. Call the number in this alert and verify your full card number.",
            "Insurance claims: EOB blocked. Upload ID and bank details through this secure message link.",
            "School portal alert: emergency form missing. Enter parent login and payment card at this link.",
        ],
    },
    {
        "category": "benign_scary_v4",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": "known sender",
        "pressure": "no sensitive pressure",
        "ask": "no sensitive ask",
        "risk_text": "none identified",
        "safe_action": "If this came from someone you know and asks for nothing sensitive, it is probably safe. Still avoid unexpected links.",
        "messages": [
            "Clinic reminder: your lab results are available in the patient portal. No urgent action is required.",
            "School alert: buses are delayed because of weather. Students are safe and pickup is at the usual place.",
            "Court reminder: jury duty starts Monday. Bring your ID and report to the address on your mailed summons.",
            "Pharmacy notice: your prescription refill is ready for pickup at your usual store.",
            "Dentist reminder: your appointment is tomorrow at 9 AM. Call the office number you already have if needed.",
            "Library notice: your requested book is ready. It will be held at the front desk until Friday.",
            "Building notice: fire alarm testing is scheduled this afternoon. No action needed.",
            "HR reminder: open enrollment ends Friday. Review benefits through the company portal.",
        ],
    },
    {
        "category": "marketplace_calibration_v4",
        "risk": "suspicious",
        "scam_type": "marketplace_scam",
        "tactics": ["marketplace", "third party", "payment pressure"],
        "impersonates": "buyer or seller",
        "pressure": "unusual transaction path",
        "ask": "leave the marketplace flow or share codes",
        "risk_text": "overpayment or account theft",
        "safe_action": "Pause before acting. Keep payment and messages inside the marketplace's official flow.",
        "messages": [
            "I want your table but I am traveling. My cousin will pick it up and I can send extra money now.",
            "Use this shipping agent and pay their small insurance deposit; I will include it in the final payment.",
            "I sent a code to prove you are real. Send it back and then I will pay.",
            "My assistant handles payment by email. Send your bank name so accounting can release funds.",
            "I can buy today, but we need to move off the marketplace because my account has a limit.",
            "The escrow site protects both of us. Please use this payment page before pickup.",
            "I accidentally overpaid. Please refund the difference to my driver.",
            "Can you mark it sold and pay the courier fee now? I promise to reimburse you.",
        ],
    },
]


def v4_cases(repeats: int) -> list[tuple[str, dict, dict]]:
    cases = []
    for repeat in range(repeats):
        for case in V4_CASES:
            prediction = hard_prediction(case)
            for index, message in enumerate(case["messages"]):
                case_id = f"{case['category']}_{repeat:02d}_{index:02d}"
                cases.append((case_id, {"message": message, "scenario": case}, prediction))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jawbreaker v4 1B reliability and boundary data.")
    parser.add_argument("--base-train", type=int, default=680)
    parser.add_argument("--base-dev", type=int, default=120)
    parser.add_argument("--base-test", type=int, default=180)
    parser.add_argument("--hard-repeats", type=int, default=4)
    parser.add_argument("--boundary-repeats", type=int, default=5)
    parser.add_argument("--v4-repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "eval" / "hard_v4_eval.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = args.base_train + args.base_dev + args.base_test
    base = list(generated_cases(total))
    v2_hard = hard_cases(args.hard_repeats)
    boundary = boundary_cases(args.boundary_repeats)
    v4 = v4_cases(args.v4_repeats)

    v2_hard_train, v2_hard_dev = split_special_cases(v2_hard, dev_fraction=0.20)
    boundary_train, boundary_dev = split_special_cases(boundary, dev_fraction=0.20)
    v4_train, v4_dev = split_special_cases(v4, dev_fraction=0.20)

    train_cases = base[: args.base_train] + v2_hard_train + boundary_train + v4_train
    dev_cases = base[args.base_train : args.base_train + args.base_dev] + v2_hard_dev + boundary_dev + v4_dev
    test_cases = base[args.base_train + args.base_dev :] + v2_hard_dev + boundary_dev + v4_dev

    write_jsonl(args.out_dir / "train_v4.jsonl", (sft_row(*case) for case in train_cases))
    write_jsonl(args.out_dir / "dev_v4.jsonl", (sft_row(*case) for case in dev_cases))
    write_jsonl(args.out_dir / "test_v4.jsonl", (sft_row(*case) for case in test_cases))
    write_jsonl(args.eval_out, (eval_row(*case) for case in test_cases))

    print(f"wrote train_v4={len(train_cases)} dev_v4={len(dev_cases)} test_v4={len(test_cases)}")
    print(f"v2_hard={len(v2_hard)} boundary={len(boundary)} v4={len(v4)}")
    print(f"v4_train={len(v4_train)} v4_dev={len(v4_dev)}")
    print(f"training data: {args.out_dir}")
    print(f"hard eval: {args.eval_out}")


if __name__ == "__main__":
    main()
