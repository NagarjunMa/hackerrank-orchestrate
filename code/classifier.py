"""
classifier.py — Rule-based escalation detection, injection detection, and company inference.

Applied BEFORE the LLM call. Deterministic safety nets that prevent the model
from trying to "help" with fraud, outages, billing disputes, security issues,
or prompt injection attacks.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Escalation patterns — any match forces status=escalated
# ---------------------------------------------------------------------------

_ESCALATION_RULES: list[tuple[str, str]] = [
    # Assessment / score fraud
    (
        r"(score|grade|result|answer).{0,40}(manipulat|chang|increas|boost|forc|hack|alter|fix)",
        "assessment fraud / score manipulation",
    ),
    (
        r"(force|make|get).{0,30}(pass|hire|accept|promot|next.{0,10}round)",
        "forced pass / hiring manipulation",
    ),
    (
        r"(unfair.{0,50}(grad|assess|evaluat|test|scor)|grad.{0,30}unfair)",
        "unfair grading dispute",
    ),
    (
        r"(cheat|bypass|circumvent).{0,30}(test|assess|system|detect)",
        "assessment integrity violation",
    ),
    (
        r"(review|show|reveal).{0,30}(my|the).{0,20}answer",
        "answer disclosure request",
    ),
    # Critical platform outages (entire service down, not a single-user issue)
    (
        r"(site|platform|website|service|system|pages?|everything|nothing).{0,30}(down|broken|not.{0,10}work|unavailable|crash|inaccessible|unreachable|failing)",
        "critical platform outage",
    ),
    (
        r"(down|broken|unavailable|crash|failing).{0,30}(site|platform|service|system|everything)",
        "critical platform outage",
    ),
    # "X is down" — catches feature-specific outages like "Resume Builder is Down"
    (
        r"\bis\s+(down|broken|not\s+working|unavailable|unreachable)\b",
        "service/feature is down",
    ),
    # "none of X are working", "all submissions failing" — systemic failure
    # Note: "all requests to [service] are failing" excluded — could be single-user API integration
    (
        r"(none\s+of|all\s+(submissions?|pages?|challenges?)).{0,50}(work|fail|access|load|respond)",
        "systemic failure / multiple submissions failing",
    ),
    # "stopped working completely" / "has stopped working"
    (
        r"(stopped|has stopped|stop).{0,20}work.{0,20}completely",
        "complete service stoppage",
    ),
    # Billing disputes — FORCED refund demands only (not general "how to" questions)
    (
        r"(force|demand|require|insist|make).{0,30}(refund|reimburse|money.{0,10}back|return.{0,10}money)",
        "forced refund demand",
    ),
    (
        r"(give me|want).{0,20}(my money|refund).{0,20}(back|now|asap|immediately|today)",
        "urgent refund demand",
    ),
    (r"\bchargeback\b", "chargeback request"),
    # Security / account compromise
    (
        r"(hack(?!errank)|compromis|breach|unauthorized).{0,30}(account|access|data|credential|password|token|login)",
        "account security breach",
    ),
    (
        r"(account|password|credential).{0,40}(hack(?!errank)|stolen|compromis|breach|leak)",
        "account security breach",
    ),
    (
        r"\bidentity.{0,10}(theft|stolen|compromis)\b",
        "identity theft",
    ),
    # Workspace / permission restoration where user has lost control
    (
        r"(restore|reinstate|give back).{0,20}(access|permission|seat|account|workspace)",
        "access restoration request",
    ),
    (
        r"(i|my).{0,10}(lost|no longer have|don.t have|cannot.{0,5}access).{0,30}(access|workspace|seat|account)",
        "lost access / seat restoration",
    ),
]

_COMPILED_ESCALATION = [
    (re.compile(pat, re.IGNORECASE | re.DOTALL), reason)
    for pat, reason in _ESCALATION_RULES
]

# ---------------------------------------------------------------------------
# Prompt injection / jailbreak patterns
# Any match → reply with invalid (not escalate, but refuse to process normally)
# ---------------------------------------------------------------------------

_INJECTION_RULES: list[tuple[str, str]] = [
    (
        r"(ignore|forget|override|disregard|bypass).{0,30}(previous|above|prior|all|your).{0,20}(instruction|rule|prompt|directive|constraint)",
        "instruction override attempt",
    ),
    (
        r"(show|display|reveal|print|output|tell me|affiche|mostra|zeige).{0,50}"
        r"(system prompt|internal rule|retrieved doc|logic you use|exact logic|instruction|context window|your prompt)",
        "system prompt extraction attempt",
    ),
    (
        r"affiche.{0,60}(règles|règle|document|logique|contexte|instruction)",
        "French prompt injection",
    ),
    (
        r"(act as|pretend (to be|you are)|you are now a|your new (role|persona|instruction))",
        "persona jailbreak",
    ),
    (r"\bjailbreak\b", "jailbreak keyword"),
    (r"\bDAN\b", "DAN jailbreak"),
    (
        r"(ignore|disregard).{0,20}(safety|ethical|guideline|policy|restriction)",
        "safety bypass attempt",
    ),
]

_COMPILED_INJECTION = [
    (re.compile(pat, re.IGNORECASE | re.DOTALL), reason)
    for pat, reason in _INJECTION_RULES
]


def check_escalation(issue: str, subject: str = "") -> tuple[bool, str]:
    """
    Check hard escalation rules.
    Returns (True, reason) if any rule fires, else (False, "").
    """
    text = f"{issue} {subject or ''}".strip()
    for pattern, reason in _COMPILED_ESCALATION:
        if pattern.search(text):
            return True, reason
    return False, ""


def check_injection(issue: str, subject: str = "") -> tuple[bool, str]:
    """
    Check for prompt injection / jailbreak attempts.
    Returns (True, reason) if detected, else (False, "").
    """
    text = f"{issue} {subject or ''}".strip()
    for pattern, reason in _COMPILED_INJECTION:
        if pattern.search(text):
            return True, reason
    return False, ""


# ---------------------------------------------------------------------------
# Company inference — keyword heuristics when Company field is None/empty
# ---------------------------------------------------------------------------

_COMPANY_KEYWORDS: dict[str, list[str]] = {
    "hackerrank": [
        "hackerrank",
        "hacker rank",
        "hackerrank for work",
        "assessment",
        "coding challenge",
        "coding test",
        "technical screen",
        "candidate",
        "recruiter",
        "hiring manager",
        "mock interview",
        "skillup",
        "engage",
        "codepair",
        "question library",
        "test variant",
        "time accommodation",
        "proctoring",
        "plagiarism",
        "test score",
        "invitation link",
        "interviewer",
        "code pair",
    ],
    "claude": [
        "claude",
        "anthropic",
        "claude.ai",
        "claude api",
        "anthropic api",
        "api key",
        "console",
        "workbench",
        "context window",
        "claude pro",
        "claude max",
        "claude team",
        "claude enterprise",
        "bedrock",
        "vertex ai",
        "claude code",
        "mcp",
        "artifacts",
        "prompt design",
        "rate limit",
        "token",
        "lti",
        "claude for education",
    ],
    "visa": [
        "visa",
        "visa card",
        "credit card",
        "debit card",
        "prepaid card",
        "merchant",
        "payment network",
        "transaction",
        "chargeback",
        "atm",
        "pin",
        "chip",
        "contactless",
        "traveller cheque",
        "travelers check",
        "lost card",
        "stolen card",
        "card stolen",
        "card lost",
        "emergency cash",
        "visa global",
        "visa support",
        "card blocked",
        "minimum spend",
    ],
}


def infer_company(issue: str, subject: str = "") -> Optional[str]:
    """
    Infer company from ticket text.
    Returns 'hackerrank', 'claude', 'visa', or None.
    """
    text = f"{issue} {subject or ''}".lower()

    scores: dict[str, int] = {c: 0 for c in _COMPANY_KEYWORDS}
    for company, keywords in _COMPANY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[company] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None
