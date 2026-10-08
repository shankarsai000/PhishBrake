from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit


_URL_PATTERN = re.compile(
    r"""(?ix)
    (?<![\w@])
    (?:(?:https?://)|(?:www\.))?
    (?P<host>
        (?:[a-z0-9\u0080-\uffff](?:[a-z0-9\u0080-\uffff-]{0,61}[a-z0-9\u0080-\uffff])?\.)+
        [a-z\u0080-\uffff]{2,63}
    )
    (?::\d{1,5})?
    (?:/[^\s<>\"']*)?
    (?:\?[^\s<>\"']*)?
    (?:\#[^\s<>\"']*)?
    """
)
_TRAILING_PUNCTUATION = ".,;:!?)]}"
_TARGET_DOMAINS = (
    "paypal.com",
    "wellsfargo.com",
    "apple.com",
    "usps.com",
    "google.com",
    "amazon.com",
    "microsoft.com",
    "chase.com",
    "bankofamerica.com",
    "fedex.com",
    "ups.com",
    "netflix.com",
    "coinbase.com",
    "instagram.com",
    "facebook.com",
    "walmart.com",
    "venmo.com",
    "zelle.com",
)


def _normalize_domain(domain: str) -> str:
    candidate = domain.strip().rstrip(".")
    if not candidate:
        return ""
    if "://" not in candidate:
        candidate = f"//{candidate}"
    try:
        return (urlsplit(candidate).hostname or "").rstrip(".").lower()
    except ValueError:
        return ""


def extract_urls(text: str) -> list[str]:
    """Extract HTTP(S), www, and bare domain URLs from a message."""
    urls: list[str] = []
    seen: set[str] = set()
    for match in _URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(_TRAILING_PUNCTUATION)
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _script_for_character(character: str) -> str | None:
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    for script in (
        "LATIN",
        "CYRILLIC",
        "GREEK",
        "ARMENIAN",
        "HEBREW",
        "ARABIC",
        "DEVANAGARI",
        "HIRAGANA",
        "KATAKANA",
        "HANGUL",
    ):
        if script in name:
            return script.title()
    if "CJK" in name or "IDEOGRAPH" in name:
        return "Han"
    return "Other"


def detect_homoglyphs(domain: str) -> list[dict]:
    """Report mixed-script labels and explicit IDNA Punycode labels."""
    normalized = _normalize_domain(domain)
    if not normalized:
        return []

    findings: list[dict] = []
    labels = normalized.split(".")
    if any(label.startswith("xn--") for label in labels):
        findings.append(
            {
                "type": "punycode",
                "domain": normalized,
                "detail": "Domain contains an IDNA Punycode label.",
            }
        )

    for label in labels:
        scripts = sorted(
            {
                script
                for character in label
                if (script := _script_for_character(character)) is not None
            }
        )
        if len(scripts) > 1:
            findings.append(
                {
                    "type": "mixed_script",
                    "domain": normalized,
                    "label": label,
                    "scripts": scripts,
                    "detail": "Domain label mixes Unicode writing systems.",
                }
            )
    return findings


def _edit_distance(left: str, right: str, max_distance: int = 2) -> int:
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        if min(current) > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def detect_typosquatting(domain: str) -> list[dict]:
    """Report close spellings and brand-prefixed deceptive domain labels."""
    normalized = _normalize_domain(domain)
    labels = normalized.split(".")
    if len(labels) < 2:
        return []

    candidate_label = labels[-2]
    candidate_tld = labels[-1]
    findings: list[dict] = []
    for target_domain in _TARGET_DOMAINS:
        target_label, target_tld = target_domain.rsplit(".", 1)
        if candidate_label == target_label and candidate_tld == target_tld:
            continue

        distance = _edit_distance(candidate_label, target_label)
        if 1 <= distance <= 2:
            findings.append(
                {
                    "type": "typosquat",
                    "domain": normalized,
                    "target": target_domain,
                    "distance": distance,
                    "detail": (
                        f"Domain label is {distance} edit(s) from {target_domain}."
                        if candidate_tld == target_tld
                        else f"Domain label is {distance} edit(s) from {target_label} on a different suffix."
                    ),
                }
            )
            continue

        if candidate_label.startswith(f"{target_label}-"):
            findings.append(
                {
                    "type": "typosquat",
                    "domain": normalized,
                    "target": target_domain,
                    "distance": None,
                    "detail": (
                        f"Domain adds a deceptive suffix to the {target_label} brand label."
                        if candidate_tld == target_tld
                        else f"Domain combines the {target_label} brand with a different suffix."
                    ),
                }
            )
    return findings


ZERO_WIDTH_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060", "\u200e", "\u200f", "\xad"}
_URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "is.gd",
    "t.co",
    "cutt.ly",
    "rb.gy",
    "ow.ly",
    "me-qr.com",
    "qr.codes",
    "goo.gl",
    "tiny.cc",
    "tr.ee",
}
_IP_HOST_PATTERN = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def strip_zero_width_chars(text: str) -> str:
    """Remove invisible zero-width and joiner characters used for evasion."""
    return "".join(ch for ch in text if ch not in ZERO_WIDTH_CHARS)


def detect_zero_width_evasion(text: str) -> bool:
    """Check if message contains zero-width unicode characters intended to evade filters."""
    return any(ch in ZERO_WIDTH_CHARS for ch in text)


def detect_url_anomalies(url: str) -> list[dict]:
    """Detect UserInfo domain spoofing (@ trick), raw IP hosts, and obfuscated link shorteners."""
    findings: list[dict] = []
    
    # UserInfo @ trick (e.g., https://paypal.com@evil.com/login)
    if "@" in url and ("http://" in url or "https://" in url):
        findings.append({
            "type": "userinfo_spoof",
            "url": url,
            "detail": "URL uses @ symbol trickery to disguise the real target domain.",
        })
        
    domain = _normalize_domain(url)
    if not domain:
        return findings

    # IP address host
    if _IP_HOST_PATTERN.match(domain):
        findings.append({
            "type": "ip_host",
            "domain": domain,
            "detail": "URL uses a raw IP address instead of a domain name.",
        })

    # URL Shortener
    if domain in _URL_SHORTENERS:
        findings.append({
            "type": "url_shortener",
            "domain": domain,
            "detail": "URL uses a link shortener to hide its final destination.",
        })

    return findings


def audit_message_urls(text: str) -> dict:
    """Extract message URLs and return spoofing findings plus prompt advisory."""
    cleaned_text = strip_zero_width_chars(text)
    urls = extract_urls(cleaned_text)
    flagged_urls: list[dict] = []
    reasons: list[dict] = []
    advisories: list[str] = []
    seen_domains: set[str] = set()

    has_zero_width = detect_zero_width_evasion(text)
    if has_zero_width:
        reasons.append({
            "type": "zero_width_evasion",
            "detail": "Message contains invisible zero-width characters intended to bypass spam filters.",
        })
        advisories.append("[System Guard: Stealth zero-width character evasion detected]")

    for url in urls:
        domain = _normalize_domain(url)
        url_reasons = detect_homoglyphs(domain) + detect_typosquatting(domain) + detect_url_anomalies(url)
        if not url_reasons:
            continue
        flagged_urls.append({"url": url, "domain": domain, "reasons": url_reasons})
        reasons.extend(url_reasons)
        if domain and domain not in seen_domains:
            seen_domains.add(domain)
            if any(r["type"] in ("mixed_script", "punycode", "typosquat") for r in url_reasons):
                advisories.append(
                    "[System Guard: Deceptive homograph/typosquat domain detected: "
                    f"{domain}]"
                )
            else:
                advisories.append(
                    "[System Guard: Deceptive or obfuscated domain detected: "
                    f"{domain}]"
                )

    return {
        "urls": urls,
        "flagged_urls": flagged_urls,
        "reasons": reasons,
        "advisory": "\n".join(advisories),
        "has_zero_width": has_zero_width,
    }

