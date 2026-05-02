"""
agent.py — Support ticket triage agent.

Orchestrates: injection check → escalation check → retrieval → LLM structured output.
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
import anthropic

from retriever import Retriever
from classifier import check_escalation, check_injection, infer_company
from prompts import SYSTEM_PROMPT, FEW_SHOT_BLOCK

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

# Tool schema — forces structured JSON output via tool_use
_RESOLVE_TOOL = {
    "name": "resolve_ticket",
    "description": "Resolve a support ticket with structured classification and response.",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["replied", "escalated"],
                "description": "Whether to reply with a corpus-grounded answer or escalate to human support.",
            },
            "product_area": {
                "type": "string",
                "description": (
                    "Support category derived from the most relevant document's breadcrumbs. "
                    "Use snake_case. Empty string if no relevant document found."
                ),
            },
            "response": {
                "type": "string",
                "description": (
                    "User-facing response. For replied: concise answer grounded in the docs. "
                    "For escalated: 'This issue requires human support. Escalating to a specialist.' "
                    "For invalid: 'I am sorry, this is out of scope from my capabilities.'"
                ),
            },
            "justification": {
                "type": "string",
                "description": "One or two sentences explaining the routing decision and which docs informed it.",
            },
            "request_type": {
                "type": "string",
                "enum": ["product_issue", "feature_request", "bug", "invalid"],
                "description": "Classification of the support request.",
            },
        },
        "required": ["status", "product_area", "response", "justification", "request_type"],
    },
}

_ESCALATION_RESPONSE = "This issue requires human support. Escalating to a specialist."
_INVALID_RESPONSE = "I am sorry, this is out of scope from my capabilities."

# Fallback when LLM call fails completely
_FALLBACK_RESULT: dict = {
    "status": "escalated",
    "product_area": "",
    "response": _ESCALATION_RESPONSE,
    "justification": "Agent failed to produce a structured response; escalating for safety.",
    "request_type": "product_issue",
}


def _validate_result(result: dict) -> dict:
    """
    Ensure all required fields are present and non-None.
    Hard-enforce enum values.
    """
    valid_statuses = {"replied", "escalated"}
    valid_request_types = {"product_issue", "feature_request", "bug", "invalid"}

    # Coerce None / missing → empty string
    for key in ("status", "product_area", "response", "justification", "request_type"):
        val = result.get(key)
        result[key] = val if (val is not None and val != "") else ""

    # Enforce valid enum values
    if result["status"] not in valid_statuses:
        result["status"] = "escalated"
    if result["request_type"] not in valid_request_types:
        result["request_type"] = "product_issue"

    # Replied tickets must have a non-empty response
    if result["status"] == "replied" and not result["response"].strip():
        result["status"] = "escalated"
        result["response"] = _ESCALATION_RESPONSE
        result["justification"] = (
            "LLM returned empty response for a replied ticket; escalating for safety."
        )

    return result


def _format_docs(docs: list[dict]) -> str:
    """Format retrieved docs into a readable context block."""
    if not docs:
        return "(No relevant documentation found.)"

    parts = []
    for i, doc in enumerate(docs, 1):
        breadcrumbs = " > ".join(doc.get("breadcrumbs") or [])
        url = doc.get("source_url", "")
        content = (doc.get("content") or "")[:3000]
        header = f"--- Document {i} | {doc['title']} ---"
        if breadcrumbs:
            header += f"\nSection: {breadcrumbs}"
        if url:
            header += f"\nSource: {url}"
        parts.append(f"{header}\n\n{content}")

    return "\n\n".join(parts)


def _build_user_message(
    issue: str,
    subject: str,
    company: Optional[str],
    docs: list[dict],
    force_escalate: bool,
    low_confidence: bool = False,
) -> str:
    """Build the full user turn: few-shots + docs + ticket."""
    context = _format_docs(docs)

    escalate_note = ""
    if force_escalate:
        escalate_note = (
            "\n⚠️  PRE-SCREENING FLAG: This ticket matched a hard escalation rule. "
            "You MUST set status=escalated regardless of the documentation.\n"
        )
    elif low_confidence:
        escalate_note = (
            "\nℹ️  LOW CORPUS CONFIDENCE: Retrieved documents have low similarity to this query. "
            "If this is an off-topic or out-of-scope request, use status=replied, request_type=invalid. "
            "Only escalate if the topic is a sensitive support matter that a human must handle.\n"
        )

    return (
        f"{FEW_SHOT_BLOCK}"
        f"Ticket:\n"
        f"  Subject: {subject or '(blank)'}\n"
        f"  Company: {company or 'Unknown'}\n"
        f"  Issue: {issue}\n"
        f"{escalate_note}"
        f"\nRelevant Documentation:\n{context}\n\n"
        f"Use the resolve_ticket tool to return your structured response."
    )


class Agent:
    def __init__(
        self,
        rebuild_index: bool = False,
        use_reranker: bool = False,
        verbose: bool = False,
    ):
        self.verbose = verbose

        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "Error: ANTHROPIC_API_KEY not set.\n"
                "  1. Copy .env.example → .env\n"
                "  2. Add your key: ANTHROPIC_API_KEY=sk-ant-...",
                file=sys.stderr,
            )
            sys.exit(1)

        self.client = anthropic.Anthropic(api_key=api_key)
        self.retriever = Retriever(rebuild=rebuild_index, use_reranker=use_reranker)

    def process(self, ticket: dict) -> dict:
        """
        Process one ticket dict (keys: Issue, Subject, Company).
        Returns dict with keys: status, product_area, response, justification, request_type.
        """
        issue = (ticket.get("Issue") or "").strip()
        subject = (ticket.get("Subject") or "").strip()
        company_raw = (ticket.get("Company") or "").strip()

        # Normalize company field: "None" / empty → None
        company: Optional[str] = None
        if company_raw and company_raw.lower() not in ("none", ""):
            company = company_raw.lower()

        # Infer company from content when not provided
        if not company:
            company = infer_company(issue, subject)

        # --- Safety gate 1: prompt injection detection ---
        is_injection, injection_reason = check_injection(issue, subject)
        if is_injection:
            if self.verbose:
                print(f"    INJECTION detected: {injection_reason}")
            return {
                "status": "replied",
                "product_area": "security",
                "response": _INVALID_RESPONSE,
                "justification": f"Prompt injection / jailbreak attempt detected: {injection_reason}.",
                "request_type": "invalid",
            }

        # --- Safety gate 2: hard escalation rules ---
        force_escalate, escalation_reason = check_escalation(issue, subject)

        # Retrieve relevant docs
        docs = self.retriever.retrieve(issue + " " + subject, company=company, top_k=5)

        # --- Safety gate 3: low retrieval confidence ---
        # Hard-escalate only when corpus gap + hard rule already fired (belt-and-suspenders).
        # For pure corpus gaps with no hard rule, hint to LLM via low_confidence flag but
        # allow it to classify off-topic queries as replied/invalid instead of escalating.
        max_score = self.retriever.max_score(docs)
        low_confidence = self.retriever.is_low_confidence(docs)
        if not force_escalate and low_confidence:
            escalation_reason = (
                f"Low retrieval confidence (score={max_score:.3f}); "
                "corpus does not cover this query."
            )

        if self.verbose:
            top_title = docs[0]["title"][:40] if docs else "none"
            print(
                f"    company={company or 'unknown'} | "
                f"force_escalate={force_escalate} | "
                f"max_score={max_score:.3f} | "
                f"top_doc={top_title!r}"
            )
            if force_escalate:
                print(f"    reason: {escalation_reason}")

        # Call LLM with full context
        # Pass low_confidence as hint — LLM may still reply/invalid for off-topic queries
        user_msg = _build_user_message(
            issue, subject, company, docs,
            force_escalate=force_escalate,
            low_confidence=low_confidence,
        )
        result = self._call_llm(user_msg)

        # --- Hard override: enforce escalation AFTER LLM call ---
        # Only hard rules trigger this override (not low-confidence alone).
        if force_escalate:
            result["status"] = "escalated"
            if not result.get("response", "").strip():
                result["response"] = _ESCALATION_RESPONSE
            if not result.get("justification", "").strip():
                result["justification"] = escalation_reason

        return _validate_result(result)

    def _call_llm(self, user_message: str) -> dict:
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0,
                system=SYSTEM_PROMPT,
                tools=[_RESOLVE_TOOL],
                tool_choice={"type": "tool", "name": "resolve_ticket"},
                messages=[{"role": "user", "content": user_message}],
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "resolve_ticket":
                    return dict(block.input)

        except anthropic.APIStatusError as exc:
            print(f"    API error {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            print(f"    API connection error: {exc}")
        except Exception as exc:
            print(f"    Unexpected error: {exc}")

        return dict(_FALLBACK_RESULT)
