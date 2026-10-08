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


V7_CALIBRATION_CASES = [
    {
        "category": "public_pattern_wrong_number_crypto_v7",
        "risk": "dangerous",
        "scam_type": "investment_scam",
        "tactics": ["wrong number", "investment grooming", "crypto request"],
        "impersonates": "friendly stranger or investor",
        "pressure": "relationship building followed by a money opportunity",
        "ask": "move money into a trading app, wallet, or investment site",
        "risk_text": "investment or crypto theft",
        "safe_action": "Do not send money or crypto. Stop the conversation and verify investment claims through a trusted advisor.",
        "messages": [
            "Sorry, is this Evelyn from the wine club? You seem kind. My uncle's trading group is opening a safe crypto slot today.",
            "Wrong number, but you are nice. I can show you how I made money on a gold trading app before the window closes.",
            "I meant to text my assistant. Since we connected, I can invite you to a private crypto lesson if you add funds first.",
            "This is awkward, wrong contact. My mentor says today's coin signal is rare. Start with a small deposit at the app link.",
            "I thought this was my cousin. I run a quiet investment group and can help you double a small wallet balance.",
            "Nice meeting you by accident. Download this trading app and I will guide your first crypto transfer.",
            "Wrong chat, but maybe fate. My aunt in finance can reserve you a high-yield currency trade if you fund it today.",
            "I sent this to the wrong number. You seem trustworthy. Want a private crypto platform invite before it closes?",
            "I am embarrassed, wrong person. My trading teacher can still add you if you move funds before tonight.",
            "You are kind for answering. I can repay that kindness with a coin signal, but you need to open the wallet first.",
            "I meant to reach my yoga friend. We are doing a short-term investment pool and one spot just opened.",
            "Wrong number. If you want, I can show you the app where my family earns daily crypto returns.",
        ],
    },
    {
        "category": "public_pattern_marketplace_money_v7",
        "risk": "dangerous",
        "scam_type": "marketplace_scam",
        "tactics": ["marketplace", "payment pressure", "code or refund request"],
        "impersonates": "buyer or seller",
        "pressure": "fast sale or pickup arrangement",
        "ask": "share a code, refund overpayment, pay courier fee, or leave the platform",
        "risk_text": "money or account theft",
        "safe_action": "Do not send codes, refunds, or pickup fees. Keep payment and messages inside the marketplace app.",
        "messages": [
            "I will buy the dresser now. I overpaid by mistake, so please refund the extra amount to my driver.",
            "The marketplace sent you a seller verification code. Read it back so I can release payment.",
            "My courier needs a refundable insurance fee before pickup. Pay it and I will add it to your total.",
            "I mailed a certified check for more than the price. Send the difference to the movers after deposit.",
            "I can pay today, but the payment page says you must buy a seller protection voucher first.",
            "My company buyer needs your email and bank name to send the official payment instructions.",
            "Please leave the marketplace chat and use this payment link so the courier can schedule pickup.",
            "I sent too much through the app. Return the extra to this wallet before my pickup person arrives.",
            "The driver is outside. Pay the pickup hold now and I will reimburse you with the item price.",
            "To prove the listing is yours, send me the six digit account code you just received.",
            "I can only complete the purchase through this escrow site. Deposit the seller fee to unlock funds.",
            "Mark the item sold before payment. My assistant will text you a bank confirmation link.",
        ],
    },
    {
        "category": "public_pattern_task_job_v7",
        "risk": "dangerous",
        "scam_type": "job_scam",
        "tactics": ["fake job", "too good to be true", "payment request"],
        "impersonates": "recruiter or task platform",
        "pressure": "easy remote income and quick onboarding",
        "ask": "pay, deposit money, use WhatsApp, or receive/forward funds",
        "risk_text": "advance fee or money mule scam",
        "safe_action": "Do not pay to start a job or move money for an employer. Verify the company through its official website.",
        "messages": [
            "Recruiter from TikTok Shop: flexible task assistant pays $480 daily. Message WhatsApp to activate the work account.",
            "Remote app tester needed. Deposit a small training balance today and withdraw commission after each task.",
            "We selected you for product optimization work. Complete prepaid tasks to unlock higher salary levels.",
            "Part-time online job: earn $650 per day. Buy the starter package and our payroll team reimburses you.",
            "Hotel review assistant role approved. Transfer the task deposit before your first assignment expires.",
            "Your remote payroll is pending. Receive customer funds in your account and forward them to our vendor.",
            "Hiring desk: equipment shipment is ready. Purchase from our approved link before training begins.",
            "Online data role pays same day. Add money to the workbench so your account can receive tasks.",
            "HR coordinator: interview is on Telegram only. Send bank login so payroll can connect your account.",
            "Task center bonus: complete ten prepaid orders and your withdrawal limit will increase.",
            "Training begins in one hour. Pay the refundable onboarding fee or your job slot closes.",
            "Payroll setup requires you to deposit our check and send the remaining balance to equipment support.",
        ],
    },
    {
        "category": "public_pattern_mfa_code_v7",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["code theft", "fake authority", "urgency"],
        "impersonates": "bank, wallet, support, or trusted service",
        "pressure": "account compromise or identity verification",
        "ask": "share a verification code, PIN, password, card, or login",
        "risk_text": "account takeover",
        "safe_action": "Do not share codes, passwords, PINs, or card numbers. Open the official app yourself.",
        "messages": [
            "Fraud desk: we are canceling a transfer. Read the one-time code you just received so we can block it.",
            "Coinbase support: phone update requested. Tell us the verification code to stop the change.",
            "Bank security: to remove the suspicious device, reply with your login code within five minutes.",
            "PayPal support: passkey added. Send the six digit code and we will secure the account.",
            "Zelle protection: payment pending. Reply with the confirmation code so we can reverse it.",
            "Card services: verify your full card number and PIN to keep the card open.",
            "Apple support: your account is locked. Share the code sent to your phone to prove ownership.",
            "Phone carrier: SIM swap in progress. Send the code now to cancel the request.",
            "Wallet security: withdrawal pending. Read the app code to freeze the wallet.",
            "Email security: login blocked. Reply with your password and recovery code for review.",
            "Credit union alert: profile change started. Call the number in this text and read the code.",
            "Marketplace support: buyer payment is waiting. Send the verification code to release funds.",
        ],
    },
    {
        "category": "public_pattern_toll_tax_benefit_v7",
        "risk": "dangerous",
        "scam_type": "government_impersonation",
        "tactics": ["fake authority", "payment request", "suspicious link"],
        "impersonates": "toll, tax, parking, benefit, or government office",
        "pressure": "fine, benefit freeze, or legal deadline",
        "ask": "pay through a message link or enter personal information",
        "risk_text": "payment or identity theft",
        "safe_action": "Do not pay through the message. Use the official agency website or a number from prior mail.",
        "messages": [
            "Toll notice: unpaid balance of $5.80 will become a $75 penalty. Pay now at http://toll-balance-pay.example.",
            "DMV final notice: registration suspended unless you verify identity and card details today.",
            "Parking bureau: citation overdue. Use this payment portal before court fees are added.",
            "Tax office: refund held for identity review. Upload SSN and bank details at this link.",
            "Benefits department: your monthly payment is paused. Confirm card and ID at http://benefit-review.example.",
            "Medicare update: coverage ends today unless you verify member number through this form.",
            "Court payment desk: missed notice fee is due by noon. Pay using the link below.",
            "State road authority: pay your unpaid toll today or vehicle registration will be blocked.",
            "IRS refund team: direct deposit failed. Re-enter your bank login before the refund expires.",
            "County clerk: property penalty unpaid. Submit payment through this urgent collection portal.",
            "Parking enforcement: final settlement offer expires tonight. Enter card details now.",
            "Transit fine office: account will be sent to collections unless you pay at the message link.",
        ],
    },
    {
        "category": "safe_everyday_family_v7",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": "known sender",
        "pressure": "ordinary household or school coordination",
        "ask": "no money, code, password, or urgent secret request",
        "risk_text": "none identified",
        "safe_action": "If this came from someone you know and asks for nothing sensitive, it is probably safe.",
        "messages": [
            "Can you pick up soup on the way home? I am making dinner around 6.",
            "School pickup moved to the side door today because of rain. No reply needed.",
            "Grandma, we are bringing dessert tonight. See you after the game.",
            "The neighbor returned your ladder and left it by the garage.",
            "Dentist reminder from Dad: appointment is on Tuesday. I added it to the calendar.",
            "Can you call me when you get home so I know you made it safely?",
            "The library book is on the kitchen table. I already renewed the other one.",
            "Soccer practice ended early. I will wait by the usual bench.",
            "Auntie says dinner moved to Sunday because the plumber is coming tomorrow.",
            "Your pharmacy bag is in the car. No payment is needed; I already handled it.",
            "The school nurse called about allergy forms. I will bring the paper copy tomorrow.",
            "Please bring the spare keys when you visit. The front door lock is sticky.",
            "Mom, the bank app says my statement is ready. I will check it later through the app.",
            "Clinic portal says my results are posted. No urgent action was listed.",
            "Package room says a box arrived. I will pick it up at the front desk.",
            "Your dentist appointment is confirmed for Tuesday at 2 PM. Reply C or call the office number on your calendar.",
        ],
    },
    {
        "category": "needs_check_official_route_v7",
        "risk": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "tactics": ["verification needed", "trusted route"],
        "impersonates": "known service or office",
        "pressure": "account, appointment, delivery, or payment notice",
        "ask": "verify through official app, official website, or known number",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Verify directly through the official app, official website, or a known phone number.",
        "messages": [
            "Bank alert: profile contact information changed. Open the official banking app yourself to review it.",
            "Credit union notice: did you attempt a $42.18 grocery purchase? Reply YES or NO, or call the number on your card.",
            "PayPal notice: a passkey was added. Go to paypal.com directly if this was not you.",
            "Coinbase notice: password changed. Open the official app yourself to review security.",
            "Carrier update: delivery delayed by weather. Track it in the official carrier app.",
            "Utility bill is ready. Pay only through your normal account or the number printed on your bill.",
            "Airline alert: flight time changed. Open the airline app directly to confirm.",
            "Hospital portal: new results are ready. Sign in through your usual patient portal.",
            "School portal: permission form deadline is Friday. Use the normal parent portal.",
            "Phone carrier: SIM protection changed. Open the carrier app directly to check account settings.",
            "Tax software: your draft return is ready. Sign in through the app you already use.",
            "Pharmacy account: order details changed. Check through your normal pharmacy login.",
        ],
    },
]


def v7_calibration_cases(repeats: int) -> list[tuple[str, dict, dict]]:
    cases = []
    for repeat in range(repeats):
        for case in V7_CALIBRATION_CASES:
            prediction = hard_prediction(case)
            for index, message in enumerate(case["messages"]):
                case_id = f"{case['category']}_{repeat:02d}_{index:02d}"
                cases.append((case_id, {"message": message, "scenario": case}, prediction))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jawbreaker v7 fresh-pattern calibration data.")
    parser.add_argument("--base-train", type=int, default=680)
    parser.add_argument("--base-dev", type=int, default=120)
    parser.add_argument("--base-test", type=int, default=180)
    parser.add_argument("--hard-repeats", type=int, default=4)
    parser.add_argument("--boundary-repeats", type=int, default=5)
    parser.add_argument("--v4-repeats", type=int, default=5)
    parser.add_argument("--v5-repeats", type=int, default=5)
    parser.add_argument("--v6-repeats", type=int, default=5)
    parser.add_argument("--v7-repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "eval" / "hard_v7_eval.jsonl")
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

    v2_hard_train, v2_hard_dev = split_special_cases(v2_hard, dev_fraction=0.20)
    boundary_train, boundary_dev = split_special_cases(boundary, dev_fraction=0.20)
    v4_train, v4_dev = split_special_cases(v4, dev_fraction=0.20)
    v5_train, v5_dev = split_special_cases(v5, dev_fraction=0.20)
    v6_train, v6_dev = split_special_cases(v6, dev_fraction=0.20)
    v7_train, v7_dev = split_special_cases(v7, dev_fraction=0.20)

    train_cases = (
        base[: args.base_train]
        + v2_hard_train
        + boundary_train
        + v4_train
        + v5_train
        + v6_train
        + v7_train
    )
    dev_cases = (
        base[args.base_train : args.base_train + args.base_dev]
        + v2_hard_dev
        + boundary_dev
        + v4_dev
        + v5_dev
        + v6_dev
        + v7_dev
    )
    test_cases = (
        base[args.base_train + args.base_dev :]
        + v2_hard_dev
        + boundary_dev
        + v4_dev
        + v5_dev
        + v6_dev
        + v7_dev
    )

    write_jsonl(args.out_dir / "train_v7.jsonl", (sft_row(*case) for case in train_cases))
    write_jsonl(args.out_dir / "dev_v7.jsonl", (sft_row(*case) for case in dev_cases))
    write_jsonl(args.out_dir / "test_v7.jsonl", (sft_row(*case) for case in test_cases))
    write_jsonl(args.eval_out, (eval_row(*case) for case in test_cases))

    print(f"wrote train_v7={len(train_cases)} dev_v7={len(dev_cases)} test_v7={len(test_cases)}")
    print(f"v2_hard={len(v2_hard)} boundary={len(boundary)} v4={len(v4)} v5={len(v5)} v6={len(v6)} v7={len(v7)}")
    print(f"v7_train={len(v7_train)} v7_dev={len(v7_dev)}")
    print(f"training data: {args.out_dir}")
    print(f"hard eval: {args.eval_out}")


if __name__ == "__main__":
    main()
