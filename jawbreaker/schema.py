from __future__ import annotations

from dataclasses import dataclass, field


RISK_LEVELS = {"dangerous", "suspicious", "needs_check", "safe"}


@dataclass
class ScamAnalysis:
    risk_level: str
    scam_type: str
    summary: str
    tactics: list[str] = field(default_factory=list)
    safest_action: str = ""
    trusted_person_message: str = ""
    scam_dna: dict[str, str] = field(default_factory=dict)
    similar_memory: str = ""

    def __post_init__(self) -> None:
        if self.risk_level not in RISK_LEVELS:
            raise ValueError(f"Invalid risk level: {self.risk_level}")

    @classmethod
    def from_heuristics(cls, message: str, memory: list[dict] | None = None) -> "ScamAnalysis":
        from jawbreaker.homograph import detect_zero_width_evasion, strip_zero_width_chars

        clean_message = strip_zero_width_chars(message)
        text = clean_message.lower()
        memory = memory or []
        has_zero_width = detect_zero_width_evasion(message)

        safe_code_notice = (
            "verification code" in text
            and "do not share this code" in text
            and "we will never ask for it" in text
            and not has_zero_width
            and not any(
                token in text
                for token in [
                    "http://",
                    "https://",
                    "reply with",
                    "send this code",
                    "enter the code",
                    "read the code",
                ]
            )
        )
        if safe_code_notice:
            return cls(
                risk_level="safe",
                scam_type="none",
                summary="This notification does not ask you to share the verification code.",
                tactics=[],
                safest_action="Do not share the code. If you did not request it, review your account through the official app.",
                trusted_person_message="Can you sanity-check this notification with me?",
                scam_dna={
                    "Impersonates": "A service sending a verification code",
                    "Pressure": "No pressure shown",
                    "Ask": "Do not share the code",
                    "Risk": "No direct scam request found",
                },
                similar_memory=_find_similar_memory(text, memory),
            )

        risk_level = "safe"
        scam_type = "none"
        tactics: list[str] = []

        if has_zero_width:
            tactics.append("zero_width_evasion")
            risk_level = "suspicious"
            scam_type = "obfuscated_message"

        credential_tokens = [
            "verify",
            "password",
            "login",
            "account locked",
            "account is locked",
            "one-time code",
            "verification code",
            "6-digit code",
            "code that just came",
            "code you just received",
            "six digit code",
            "passkey",
            "read the code",
            "read it back",
            "read the app code",
            "pin",
            "card number",
            "full card number",
            "payment card",
            "bank login",
            "bank details",
            "seed phrase",
            "username",
            "parent login",
            "phone number on your account",
            "send your details",
            "account details",
            "upload id",
            "insurance number",
            "card details",
            "confirm ownership",
            "confirm your information",
            "confirm your billing info",
            "account reset code",
            "reset code",
            "unauthorized login",
            "did you request a password reset",
            "did you authorize",
            "zelle transfer attempt",
        ]
        if any(token in text for token in credential_tokens):
            risk_level = "dangerous"
            scam_type = "credential_theft"
            tactics.extend(["credential request", "fake authority"])

        if any(token in text for token in ["urgent", "immediately", "24 hours", "act now", "held"]):
            if risk_level == "safe":
                risk_level = "suspicious"
            tactics.append("urgency")

        payment_tokens = [
            "gift card",
            "gift cards",
            "apple cards",
            "google play cards",
            "prepaid cards",
            "prepaid card",
            "zelle",
            "crypto",
            "wire",
            "$800",
            "send money",
            "send funds",
            "processing fee",
            "onboarding fee",
            "training fee",
            "cleanup fee",
            "delivery tax",
            "past-due tax",
            "unpaid balance",
            "registration suspended",
            "citation overdue",
            "refund held",
            "monthly payment is paused",
            "missed notice fee",
            "property penalty",
            "vehicle registration",
            "pay the",
            "pay an",
            "buy equipment",
            "deposit a check",
            "deposit this check",
            "$300",
            "$500",
            "hospital bill",
            "paying a bill",
            "insurance payment",
            "small insurance payment",
            "reimburse you",
            "service fee",
            "troubleshooting voucher",
            "courier fee",
            "refund the difference",
            "payment app transfer",
            "customer funds",
            "forward them to our vendor",
            "unpaid toll",
            "toll road invoice",
            "ezpass fee",
            "sunpass invoice",
            "redelivery fee",
            "customs fee",
        ]
        if any(token in text for token in payment_tokens):
            risk_level = "dangerous"
            scam_type = "payment_request"
            tactics.append("payment pressure")

        if any(token in text for token in ["scan the qr", "scan qr code", "scan to pay", "qr code below"]):
            risk_level = "dangerous"
            scam_type = "credential_theft"
            tactics.extend(["quishing_qr_prompt", "suspicious link"])

        family_terms = ["grandma", "grandpa", "auntie", "niece", "nephew", "your daughter"]
        number_change_terms = ["new number", "changed numbers", "number only", "phone broke", "temporary phone"]
        secrecy_terms = ["don't tell", "dont tell", "do not tell", "keep this quiet", "keep it quiet"]
        family_pressure_terms = [
            "wire",
            "gift card",
            "send money",
            "send funds",
            "rent",
            "hospital bill",
            "bail",
            "emergency",
            "lost my phone",
            "need you to",
            "can you send",
        ]
        family_context = any(token in text for token in family_terms)
        number_change = any(token in text for token in number_change_terms)
        secrecy = any(token in text for token in secrecy_terms)
        family_pressure = any(token in text for token in family_pressure_terms)
        if number_change or secrecy or (family_context and family_pressure):
            risk_level = "dangerous"
            scam_type = "family_impersonation"
            tactics.append("impersonation")
            if secrecy:
                tactics.append("secrecy")
            if family_pressure:
                tactics.append("payment pressure")

        if any(
            token in text
            for token in [
                "remote support",
                "remote access",
                "screen code",
                "install remote",
                "technician",
                "microsoft security",
                "apple support",
                "browser warning",
                "geek help desk",
                "refund tool",
                "call this number",
                "call the support number",
                "call support",
                "number shown",
                "callback number",
                "call the number in this alert",
                "call the number in this message",
                "call the number in this text",
                "call the number shown",
                "call the listed support number",
                "device protection",
                "subscription expired",
                "avoid data loss",
                "data loss",
                "desktop access code",
                "remote cleanup tool",
                "remote access software",
            ]
        ):
            risk_level = "dangerous"
            scam_type = "tech_support"
            tactics.extend(["fake authority", "remote access request"])

        if any(token in text for token in ["coinbase", "crypto account", "account update"]) and any(
            token in text for token in ["call support", "callback number", "contact support immediately"]
        ):
            risk_level = "dangerous"
            scam_type = "callback_phishing"
            tactics.extend(["fake authority", "callback request", "account takeover"])

        if any(token in text for token in ["prize", "lottery", "sweepstakes", "winner selected", "government grant"]):
            risk_level = "dangerous"
            scam_type = "prize_scam"
            tactics.extend(["too good to be true", "fake authority"])

        if any(
            token in text
            for token in [
                "you are hired",
                "job offer",
                "remote job approved",
                "equipment vendor",
                "task assistant",
                "training balance",
                "prepaid tasks",
                "starter package",
                "task deposit",
                "task center bonus",
                "workbench",
                "work account",
                "prepaid orders",
                "refundable onboarding fee",
                "task work",
                "training starts",
                "equipment shipment",
                "payroll setup",
                "recruiter",
                "part-time assistant",
                "tiktok shop",
                "whatsapp",
                "$720",
            ]
        ):
            risk_level = "dangerous"
            scam_type = "job_scam"
            tactics.extend(["fake job", "fake authority"])
            if any(token in text for token in ["per day", "$330", "$750", "payment is made immediately"]):
                tactics.append("too good to be true")

        if any(token in text for token in ["feel so close", "about us", "we can meet", "wallet was stolen"]):
            risk_level = "dangerous"
            scam_type = "romance_scam"
            tactics.extend(["emotional manipulation", "payment pressure"])

        wrong_contact_terms = [
            "wrong number",
            "wrong chat",
            "wrong contact",
            "wrong person",
            "wrong recipient",
            "texted the wrong",
            "meant to text",
            "meant to reach",
            "saved the wrong",
            "thought this was",
            "typed one digit wrong",
        ]
        investment_terms = [
            "investment",
            "investment pool",
            "investment group",
            "trading",
            "trading group",
            "trading app",
            "gold futures",
            "gold strategy",
            "currency signal",
            "coin signal",
            "crypto",
            "coin group",
            "wallet",
            "exchange app",
            "exchange account",
            "token round",
            "private exchange",
            "guaranteed returns",
            "low-risk",
            "daily returns",
            "one spot just opened",
            "reserve you a spot",
            "window closes",
            "fund the account",
            "small balance",
            "small deposit",
            "app link",
            "open the app",
        ]
        if any(token in text for token in wrong_contact_terms):
            if any(token in text for token in investment_terms):
                risk_level = "dangerous"
                scam_type = "investment_scam"
                tactics.extend(["wrong number", "relationship building", "investment bait"])
                if any(token in text for token in ["wallet", "exchange", "fund the account", "small balance"]):
                    tactics.append("wallet request")
            elif risk_level == "safe":
                risk_level = "suspicious"
                scam_type = "unknown_contact"
                tactics.extend(["wrong number", "relationship building"])

        if any(
            token in text
            for token in [
                "mover will pick it up",
                "shipping agent",
                "escrow service",
                "escrow site",
                "courier needs",
                "courier fee",
                "assistant will handle payment",
                "email your bank name",
                "extra money",
                "overpaid",
                "move off the marketplace",
                "code to prove",
                "marketplace",
                "respond quickly",
                "paperwork is missing",
            ]
        ):
            if risk_level == "safe":
                risk_level = "suspicious"
                scam_type = "marketplace_scam"
            tactics.extend(["marketplace", "payment pressure"])

        if "http://" in text or "https://" in text:
            if risk_level == "safe":
                risk_level = "suspicious"
            tactics.append("suspicious link")
            if any(
                token in text
                for token in [
                    "payment failed",
                    "payment method",
                    "update billing",
                    "billing",
                    "account closes",
                    "secure your account",
                    "enter the code",
                    "payment card",
                    "parent login",
                    "card details",
                ]
            ):
                risk_level = "dangerous"
                scam_type = "credential_theft"
                tactics.extend(["credential request", "fake authority"])

        if "reply yes or no" in text and "fraud alert" in text:
            risk_level = "needs_check"
            scam_type = "possible_legitimate_alert"
            tactics = ["verification needed"]

        tactics = sorted(set(tactics))
        summary = _summary_for(risk_level, scam_type)
        similar_memory = _find_similar_memory(text, memory)

        return cls(
            risk_level=risk_level,
            scam_type=scam_type,
            summary=summary,
            tactics=tactics,
            safest_action=_safe_action_for(risk_level, scam_type),
            trusted_person_message=_trusted_message_for(risk_level, scam_type),
            scam_dna={
                "Impersonates": _guess_impersonation(text),
                "Pressure": _guess_pressure(tactics),
                "Ask": _guess_ask(text, scam_type),
                "Risk": scam_type.replace("_", " "),
            },
            similar_memory=similar_memory,
        )


def _summary_for(risk_level: str, scam_type: str) -> str:
    if risk_level == "dangerous":
        return f"This looks dangerous: likely {scam_type.replace('_', ' ')}."
    if risk_level == "suspicious":
        return "This has warning signs and should be checked before you act."
    if risk_level == "needs_check":
        return "This might be legitimate, but you should verify it using a trusted route."
    return "No strong scam pattern was found in this short scan."


def _safe_action_for(risk_level: str, scam_type: str) -> str:
    if risk_level == "dangerous":
        return "Do not click links, do not reply, and do not send money. Contact the company or person using a number or app you already trust."
    if risk_level == "suspicious":
        return "Pause before acting. Open the official website or app yourself instead of using links from this message."
    if risk_level == "needs_check":
        return "Verify directly through the official app, official website, or a known phone number."
    return "If this came from someone you know and it asks for nothing sensitive, it is probably safe. Still avoid unexpected links."


def _trusted_message_for(risk_level: str, scam_type: str) -> str:
    if risk_level == "safe":
        return "Can you sanity-check this message for me? PhishBrake did not find a strong scam pattern, but I want to be careful."
    return f"Can you check this for me? PhishBrake says it may be {scam_type.replace('_', ' ')} and recommends that I do not click or reply yet."


def _guess_impersonation(text: str) -> str:
    if "usps" in text:
        return "USPS or package carrier"
    if "chase" in text or "bank" in text:
        return "Bank or financial institution"
    if "coinbase" in text:
        return "Coinbase or crypto platform"
    if any(token in text for token in ["grandma", "grandpa", "niece", "new number", "changed numbers"]):
        return "Family member"
    if "support" in text or "technician" in text:
        return "Tech support"
    if "recruiter" in text or "hiring" in text or "you are hired" in text or "tiktok shop" in text:
        return "Employer or recruiter"
    if any(token in text for token in ["trading", "investment", "crypto", "wallet", "exchange app"]):
        return "Friendly stranger or investment contact"
    if "prize" in text or "lottery" in text or "grant" in text:
        return "Prize or grant office"
    return "Unknown sender"


def _guess_pressure(tactics: list[str]) -> str:
    if "urgency" in tactics:
        return "Act now"
    if "secrecy" in tactics:
        return "Keep it secret"
    if "payment pressure" in tactics:
        return "Send money"
    return "Not obvious"


def _guess_ask(text: str, scam_type: str) -> str:
    if scam_type == "credential_theft":
        return "Verify account or login"
    if scam_type == "payment_request":
        return "Send money"
    if scam_type == "family_impersonation":
        return "Trust a new number"
    if scam_type == "tech_support":
        return "Install software or share access"
    if scam_type == "prize_scam":
        return "Pay or share details to claim prize"
    if scam_type == "job_scam":
        return "Pay or move money for job"
    if scam_type == "investment_scam":
        return "Join an investment or trading group"
    if scam_type == "callback_phishing":
        return "Call a number from the message"
    if "http://" in text or "https://" in text:
        return "Open a link"
    return "No direct ask found"


def _find_similar_memory(text: str, memory: list[dict]) -> str:
    tokens = set(text.split())
    best_score = 0.0
    best = ""
    for item in memory:
        old_tokens = set(str(item.get("text", "")).lower().split())
        if not old_tokens:
            continue
        score = len(tokens & old_tokens) / max(len(tokens | old_tokens), 1)
        if score > best_score:
            best_score = score
            best = str(item.get("summary", "a previous saved scam"))
    if best_score >= 0.18:
        return f"This resembles a saved pattern: {best}"
    return ""
