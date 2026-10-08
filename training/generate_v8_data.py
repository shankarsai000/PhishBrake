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


V8_CALIBRATION_CASES = [
    {
        "category": "wrong_number_investment_danger_v8",
        "risk": "dangerous",
        "scam_type": "investment_scam",
        "tactics": ["wrong number", "relationship building", "investment bait", "wallet request"],
        "impersonates": "friendly stranger or investment mentor",
        "pressure": "accidental contact becomes a private investment opportunity",
        "ask": "open a wallet, exchange, trading app, or private investment group",
        "risk_text": "investment or crypto theft",
        "safe_action": "Do not open trading apps, wallets, or investment links from a stranger. Stop the conversation.",
        "messages": [
            "Wrong contact, sorry. I work in gold futures and there is a low-risk entry tonight if you create a wallet first.",
            "I meant to text my driver. Since you seem kind, I can show a short-term coin plan through my exchange app.",
            "This is the wrong number, but maybe lucky timing. My mentor's trading group has guaranteed returns this week.",
            "Sorry, I saved the wrong contact. If you use exchange accounts, I can guide you into a private token round.",
            "I thought this was Lena. My aunt teaches a gold strategy and can reserve you a spot if you open the app.",
            "Wrong chat. You are polite, so I can invite you to a quiet investment group before the window closes.",
            "I texted the wrong person. My trading teacher says today's currency signal is safe if you fund the account.",
            "Sorry for the mistake. I can help you earn through a wallet app if you start with a small balance.",
            "Wrong number, my fault. My family uses a private exchange with daily returns, and I can introduce you.",
            "This is embarrassing. I thought you were Daniel from yoga. My mentor can still add you to the coin group.",
            "I did not mean to bother you. If you want, I can walk you through a gold futures account tonight.",
            "Wrong recipient. A low-risk crypto entry is open for a few hours, but you need the trading app first.",
        ],
    },
    {
        "category": "wrong_number_social_no_money_v8",
        "risk": "suspicious",
        "scam_type": "unknown_contact",
        "tactics": ["wrong number", "relationship building"],
        "impersonates": "unknown sender",
        "pressure": "unexpected personal conversation",
        "ask": "continue chatting without a clear trusted relationship",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Be cautious with an unknown sender. Do not share personal, financial, or account details.",
        "messages": [
            "Sorry, wrong number. I was trying to reach Anna from the volunteer group.",
            "I think I saved the wrong contact. Hope I did not bother you.",
            "Wrong person, sorry. Have a good day.",
            "This is not Maya? My mistake, I copied the number wrong.",
            "Sorry about that, I meant to text my cousin.",
            "Wrong chat. Thanks for letting me know.",
            "I must have typed one digit wrong. Sorry for the confusion.",
            "Not the person I expected. My apologies.",
            "Wrong number, but thanks for being kind.",
            "Sorry, I was trying to confirm a dinner plan with someone else.",
        ],
    },
    {
        "category": "safe_family_logistics_v8",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": "known family member",
        "pressure": "ordinary family coordination",
        "ask": "no money, code, secrecy, password, or urgent payment",
        "risk_text": "none identified",
        "safe_action": "If this came from someone you know and asks for nothing sensitive, it is probably safe.",
        "messages": [
            "Hi Grandma, we are bringing chili around 6. Do not cook tonight.",
            "Grandma, I picked up the soup and will be there after work.",
            "Mom says dinner is at 6:30. No need to bring anything.",
            "Auntie, we are dropping off leftovers before the game.",
            "Dad, I left the library book on the kitchen counter.",
            "Can you bring the spare blanket when you come over Sunday?",
            "We are running ten minutes late for dinner, but everything is fine.",
            "Grandpa, the kids made cookies and want to bring some over.",
            "I put your mail on the table. Nothing urgent.",
            "We are bringing groceries after school pickup.",
            "Dinner moved to Sunday because the plumber is coming tomorrow.",
            "I found your glasses in the car and will bring them tonight.",
            "Can you pick up milk on the way home? No rush.",
            "I watered the plants and locked the back door.",
            "Soup is in the fridge. Heat it for two minutes.",
            "We will come by at 6 with dessert. You do not need to cook.",
        ],
    },
    {
        "category": "safe_school_pickup_v8",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": "school or known caregiver",
        "pressure": "routine schedule update",
        "ask": "bring usual ID or follow normal pickup process",
        "risk_text": "none identified",
        "safe_action": "If this matches your normal school routine, it is probably safe. Use the school office number if unsure.",
        "messages": [
            "School reminder: early pickup is at 1:45 PM today. Bring your usual ID to the front office.",
            "School office: pickup is at the side door because of rain. Staff will check regular IDs.",
            "Reminder: parent-teacher pickup starts at 2 PM in the normal office.",
            "The bus is running ten minutes late. Students will wait at the usual stop.",
            "School nurse: allergy forms can be returned tomorrow with the regular paperwork.",
            "After-school club ends at 4 today. Pickup is at the usual door.",
            "Weather update: soccer practice is cancelled. Students should go home as usual.",
            "Field trip reminder: bring the signed paper form tomorrow morning.",
            "Early dismissal today at 1:30. Bring the same pickup ID you normally use.",
            "School portal: report cards are posted. Use the regular parent portal.",
            "Office reminder: lunch account notices go home in backpacks today.",
            "The choir concert starts at 6 PM. Enter through the main school doors.",
        ],
    },
    {
        "category": "safe_pharmacy_clinic_v8",
        "risk": "safe",
        "scam_type": "none",
        "tactics": [],
        "impersonates": "pharmacy or clinic",
        "pressure": "routine health notice",
        "ask": "use known store, bottle, calendar, or patient portal route",
        "risk_text": "none identified",
        "safe_action": "If this matches your usual clinic or pharmacy, use your normal portal or known number.",
        "messages": [
            "Your regular pharmacy says one prescription is ready. Use the store number printed on your bottle if you have questions.",
            "Clinic reminder: your appointment is tomorrow at 9. Use the patient portal you normally use.",
            "Dentist reminder: appointment Tuesday at 2 PM. Call the office number on your calendar if needed.",
            "Pharmacy update: one refill is ready at your usual store.",
            "Clinic portal says lab results are posted. No urgent action was listed.",
            "Vet clinic: vaccine records are available in the usual pet portal.",
            "Hospital portal: new visit summary is ready. Use your normal login.",
            "Pharmacy order shipped. Track it through your normal pharmacy account.",
            "Clinic reminder: fasting is recommended before tomorrow's appointment.",
            "Eye doctor reminder: appointment is confirmed for Friday morning.",
            "Prescription pickup is ready. No payment is due in this message.",
            "Dental office: cleaning reminder for next week. Call the office number you already have.",
        ],
    },
    {
        "category": "school_clinic_payment_link_danger_v8",
        "risk": "dangerous",
        "scam_type": "credential_theft",
        "tactics": ["fake authority", "payment request", "suspicious link", "credential request"],
        "impersonates": "school, clinic, or pharmacy",
        "pressure": "urgent access or service problem",
        "ask": "enter card, login, insurance, or identity details through a message link",
        "risk_text": "payment or identity theft",
        "safe_action": "Do not use the message link. Contact the school, clinic, or pharmacy through a number you already trust.",
        "messages": [
            "School portal emergency: pay the missing lunch fee at http://school-fee-review.example or pickup is blocked.",
            "Clinic notice: lab results are blocked until you upload ID and insurance at this link.",
            "Pharmacy alert: prescription cancelled unless you enter card details at http://rx-release.example today.",
            "School office: verify parent login and payment card now to keep pickup privileges.",
            "Dentist billing: appointment cancelled unless you pay the reschedule fee through this message link.",
            "Hospital portal: confirm SSN and bank card at the secure form to unlock results.",
            "School forms desk: enter your parent password at http://parent-form-update.example before noon.",
            "Pharmacy delivery: address incomplete. Pay a small redelivery fee at this link.",
            "Clinic support: call the number in this text and read your patient portal code.",
            "School account: device login failed. Send the verification code so we can restore parent access.",
        ],
    },
    {
        "category": "official_route_needs_check_v8",
        "risk": "needs_check",
        "scam_type": "possible_legitimate_alert",
        "tactics": ["verification needed", "trusted route"],
        "impersonates": "known service or office",
        "pressure": "routine account, delivery, payment, or health notice",
        "ask": "verify through official app, official website, or known number",
        "risk_text": "uncertain legitimacy",
        "safe_action": "Verify directly through the official app, official website, or a known phone number.",
        "messages": [
            "Credit union alert: did you attempt a $39.44 pharmacy purchase? Reply YES or NO, or call the number on your card.",
            "Delivery carrier: address needs review. Open the official carrier app yourself to check.",
            "PayPal security: new login detected. Go to paypal.com directly if this was not you.",
            "Bank notice: profile details changed. Open the official banking app yourself.",
            "Utility account: your statement is ready. Pay only through your normal account.",
            "Phone carrier: SIM protection changed. Open the carrier app directly to review.",
            "County tax notice is available. Use the official county website from prior mail.",
            "Insurance portal: explanation of benefits is ready in the normal member portal.",
            "School portal: permission form deadline is Friday. Use the usual parent portal.",
            "Pharmacy account: order details changed. Check through your normal pharmacy login.",
            "Airline alert: flight time changed. Open the airline app directly.",
            "Hospital portal: new results are ready. Sign in through your usual patient portal.",
        ],
    },
]


def v8_calibration_cases(repeats: int) -> list[tuple[str, dict, dict]]:
    cases = []
    for repeat in range(repeats):
        for case in V8_CALIBRATION_CASES:
            prediction = hard_prediction(case)
            for index, message in enumerate(case["messages"]):
                case_id = f"{case['category']}_{repeat:02d}_{index:02d}"
                cases.append((case_id, {"message": message, "scenario": case}, prediction))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jawbreaker v8 failure-driven calibration data.")
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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training" / "data")
    parser.add_argument("--eval-out", type=Path, default=ROOT / "eval" / "hard_v8_eval.jsonl")
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

    v2_hard_train, v2_hard_dev = split_special_cases(v2_hard, dev_fraction=0.20)
    boundary_train, boundary_dev = split_special_cases(boundary, dev_fraction=0.20)
    v4_train, v4_dev = split_special_cases(v4, dev_fraction=0.20)
    v5_train, v5_dev = split_special_cases(v5, dev_fraction=0.20)
    v6_train, v6_dev = split_special_cases(v6, dev_fraction=0.20)
    v7_train, v7_dev = split_special_cases(v7, dev_fraction=0.20)
    v8_train, v8_dev = split_special_cases(v8, dev_fraction=0.20)

    train_cases = (
        base[: args.base_train]
        + v2_hard_train
        + boundary_train
        + v4_train
        + v5_train
        + v6_train
        + v7_train
        + v8_train
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
    )

    write_jsonl(args.out_dir / "train_v8.jsonl", (sft_row(*case) for case in train_cases))
    write_jsonl(args.out_dir / "dev_v8.jsonl", (sft_row(*case) for case in dev_cases))
    write_jsonl(args.out_dir / "test_v8.jsonl", (sft_row(*case) for case in test_cases))
    write_jsonl(args.eval_out, (eval_row(*case) for case in test_cases))

    print(f"wrote train_v8={len(train_cases)} dev_v8={len(dev_cases)} test_v8={len(test_cases)}")
    print(
        f"v2_hard={len(v2_hard)} boundary={len(boundary)} v4={len(v4)} "
        f"v5={len(v5)} v6={len(v6)} v7={len(v7)} v8={len(v8)}"
    )
    print(f"v8_train={len(v8_train)} v8_dev={len(v8_dev)}")
    print(f"training data: {args.out_dir}")
    print(f"hard eval: {args.eval_out}")


if __name__ == "__main__":
    main()
