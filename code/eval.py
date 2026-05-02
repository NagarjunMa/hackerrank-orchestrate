"""
eval.py — Evaluate the agent against sample_support_tickets.csv (which has ground truth).

Scoring:
  status       — exact match (replied / escalated)
  request_type — exact match (product_issue / feature_request / bug / invalid)
  product_area — fuzzy match (normalize case + underscores, check substring)
  response     — LLM judge: hallucination + completeness (0.0 – 1.0)
  overall      — average of all four field scores

Usage:
    python code/eval.py                        # full eval with LLM judge
    python code/eval.py --no-llm-judge        # skip LLM judge (faster, offline)
    python code/eval.py --verbose             # show per-ticket details
    python code/eval.py --limit 5            # evaluate only first N tickets
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass

# Make code/ importable
sys.path.insert(0, str(Path(__file__).parent))

from agent import Agent

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_CSV = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"
EVAL_OUTPUT = REPO_ROOT / "support_tickets" / "eval_output.csv"

# ── LLM Judge ───────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are an objective evaluator of AI support agent responses.
You will be given:
  - The support ticket (issue + subject)
  - The expected response (ground truth from human agent)
  - The predicted response (from the AI agent)
  - The retrieved documentation used to generate the predicted response

Score the predicted response on two dimensions, each 0.0 to 1.0:

hallucination_score:
  1.0 = predicted response contains ONLY information from the retrieved documents or is a valid non-answer (escalation/invalid).
  0.5 = predicted response contains minor details not in documents but core answer is corpus-grounded.
  0.0 = predicted response fabricates policies, steps, phone numbers, or URLs not in documents.

completeness_score:
  1.0 = predicted response fully addresses the user's issue.
  0.5 = predicted response partially addresses the issue (missing steps or details).
  0.0 = predicted response does not address the issue at all (wrong topic, empty, etc.).

Return a JSON object only:
{"hallucination_score": <float>, "completeness_score": <float>, "notes": "<brief reason>"}
"""


def llm_judge(
    issue: str,
    subject: str,
    expected_response: str,
    predicted_response: str,
    retrieved_docs_context: str,
    client,
) -> dict:
    """Use Claude to score response quality. Returns {hallucination_score, completeness_score, notes}."""
    user_msg = f"""Ticket:
  Subject: {subject}
  Issue: {issue}

Expected response (ground truth):
{expected_response}

Predicted response (AI agent):
{predicted_response}

Retrieved documentation used:
{retrieved_docs_context[:2000]}

Score the predicted response. Return JSON only."""

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            temperature=0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = resp.content[0].text.strip()
        # Extract JSON from response
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as exc:
        print(f"      Judge error: {exc}")

    return {"hallucination_score": -1.0, "completeness_score": -1.0, "notes": "judge_failed"}


# ── Scoring helpers ──────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    """Normalize for fuzzy comparison: lowercase, strip, replace spaces/hyphens with underscore."""
    return re.sub(r"[\s\-]+", "_", (s or "").lower().strip())


def score_exact(predicted: str, expected: str) -> float:
    return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0


def score_product_area(predicted: str, expected: str) -> float:
    p = normalize(predicted)
    e = normalize(expected)
    if not e:
        return 1.0  # no ground truth to compare against
    if p == e:
        return 1.0
    # Substring match (e.g. "screen" in "hackerrank_screen")
    if p in e or e in p:
        return 0.8
    # First token match (e.g. both start with "screen")
    if p.split("_")[0] == e.split("_")[0]:
        return 0.6
    return 0.0


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def fmt_score(score: float) -> str:
    if score >= 0.8:
        return color(f"{score:.2f}", "32")   # green
    if score >= 0.5:
        return color(f"{score:.2f}", "33")   # yellow
    return color(f"{score:.2f}", "31")       # red


# ── Load sample CSV ──────────────────────────────────────────────────────────

def load_sample(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


# ── Main eval loop ───────────────────────────────────────────────────────────

def run_eval(args) -> None:
    if not SAMPLE_CSV.exists():
        print(f"Error: sample CSV not found: {SAMPLE_CSV}")
        sys.exit(1)

    samples = load_sample(SAMPLE_CSV)
    if args.limit:
        samples = samples[: args.limit]

    print(f"Evaluating {len(samples)} sample tickets...")

    agent = Agent(verbose=False)

    # For LLM judge, we need the Anthropic client
    judge_client = agent.client if not args.no_llm_judge else None

    results = []
    field_scores = {"status": [], "request_type": [], "product_area": [], "response": []}

    for i, sample in enumerate(samples, 1):
        issue = sample.get("Issue", "")
        subject = sample.get("Subject", "")
        company = sample.get("Company", "")

        expected_status = (sample.get("Status") or "").strip().lower()
        expected_request_type = (sample.get("Request Type") or "").strip().lower().replace(" ", "_")
        expected_product_area = (sample.get("Product Area") or "").strip()
        expected_response = (sample.get("Response") or "").strip()

        ticket = {"Issue": issue, "Subject": subject, "Company": company}
        predicted = agent.process(ticket)

        pred_status = predicted.get("status", "")
        pred_request_type = predicted.get("request_type", "")
        pred_product_area = predicted.get("product_area", "")
        pred_response = predicted.get("response", "")
        pred_justification = predicted.get("justification", "")

        # Score categorical fields
        s_status = score_exact(pred_status, expected_status)
        s_request_type = score_exact(pred_request_type, expected_request_type)
        s_product_area = score_product_area(pred_product_area, expected_product_area)

        # Score response quality
        s_hallucination = -1.0
        s_completeness = -1.0
        judge_notes = ""

        if judge_client:
            # Get the docs context for the judge
            retrieved = agent.retriever.retrieve(issue + " " + subject, company=company.lower() if company and company.lower() != "none" else None, top_k=5)
            docs_context = "\n".join(f"{d['title']}: {d['content'][:300]}" for d in retrieved)

            judge_result = llm_judge(
                issue, subject, expected_response, pred_response, docs_context, judge_client
            )
            s_hallucination = judge_result.get("hallucination_score", -1.0)
            s_completeness = judge_result.get("completeness_score", -1.0)
            judge_notes = judge_result.get("notes", "")

            # Composite response score: hallucination more important than completeness
            if s_hallucination >= 0 and s_completeness >= 0:
                s_response = 0.6 * s_hallucination + 0.4 * s_completeness
            else:
                s_response = -1.0
        else:
            s_response = -1.0

        # Overall (exclude response if judge not run)
        scored = [s_status, s_request_type, s_product_area]
        if s_response >= 0:
            scored.append(s_response)
        overall = sum(scored) / len(scored)

        field_scores["status"].append(s_status)
        field_scores["request_type"].append(s_request_type)
        field_scores["product_area"].append(s_product_area)
        if s_response >= 0:
            field_scores["response"].append(s_response)

        # Print per-ticket results
        subject_short = subject[:40] or "(no subject)"
        print(f"\n[{i:>2}] {subject_short}")
        print(f"  Issue: {issue[:80]}")
        print(f"  Company: {company}")
        print(
            f"  status      exp={expected_status:<10} pred={pred_status:<10}  {fmt_score(s_status)}"
        )
        print(
            f"  req_type    exp={expected_request_type:<15} pred={pred_request_type:<15}  {fmt_score(s_request_type)}"
        )
        print(
            f"  product_area exp={expected_product_area:<20} pred={pred_product_area:<20}  {fmt_score(s_product_area)}"
        )
        if s_response >= 0:
            print(f"  response    hallucination={s_hallucination:.2f}  completeness={s_completeness:.2f}  {fmt_score(s_response)}")
            if judge_notes:
                print(f"  judge notes: {judge_notes}")
        print(f"  overall: {fmt_score(overall)}")

        if args.verbose:
            print(f"  predicted response: {pred_response[:200]}")
            print(f"  expected  response: {expected_response[:200]}")
            print(f"  justification: {pred_justification[:150]}")

        results.append(
            {
                "issue": issue,
                "subject": subject,
                "company": company,
                "expected_status": expected_status,
                "predicted_status": pred_status,
                "score_status": s_status,
                "expected_request_type": expected_request_type,
                "predicted_request_type": pred_request_type,
                "score_request_type": s_request_type,
                "expected_product_area": expected_product_area,
                "predicted_product_area": pred_product_area,
                "score_product_area": s_product_area,
                "score_hallucination": s_hallucination,
                "score_completeness": s_completeness,
                "score_response": s_response,
                "overall": overall,
                "predicted_response": pred_response,
                "expected_response": expected_response,
                "justification": pred_justification,
                "judge_notes": judge_notes,
            }
        )

    # ── Aggregate summary ────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)

    agg_status = sum(field_scores["status"]) / len(field_scores["status"]) if field_scores["status"] else 0
    agg_rt = sum(field_scores["request_type"]) / len(field_scores["request_type"]) if field_scores["request_type"] else 0
    agg_pa = sum(field_scores["product_area"]) / len(field_scores["product_area"]) if field_scores["product_area"] else 0
    agg_resp = sum(field_scores["response"]) / len(field_scores["response"]) if field_scores["response"] else -1

    print(f"  status       accuracy: {fmt_score(agg_status)}  ({sum(1 for s in field_scores['status'] if s == 1.0)}/{len(field_scores['status'])} exact)")
    print(f"  request_type accuracy: {fmt_score(agg_rt)}  ({sum(1 for s in field_scores['request_type'] if s == 1.0)}/{len(field_scores['request_type'])} exact)")
    print(f"  product_area accuracy: {fmt_score(agg_pa)}")
    if agg_resp >= 0:
        print(f"  response quality:      {fmt_score(agg_resp)}")

    scored_agg = [agg_status, agg_rt, agg_pa]
    if agg_resp >= 0:
        scored_agg.append(agg_resp)
    overall_agg = sum(scored_agg) / len(scored_agg)
    print(f"\n  OVERALL SCORE: {fmt_score(overall_agg)}")

    # ── Error analysis ────────────────────────────────────────────────────────

    print("\n--- ERRORS ---")
    for r in results:
        errors = []
        if r["score_status"] < 1.0:
            errors.append(f"status: exp={r['expected_status']} pred={r['predicted_status']}")
        if r["score_request_type"] < 1.0:
            errors.append(f"request_type: exp={r['expected_request_type']} pred={r['predicted_request_type']}")
        if r["score_product_area"] < 0.8:
            errors.append(f"product_area: exp={r['expected_product_area']} pred={r['predicted_product_area']}")
        if r.get("score_response", -1) >= 0 and r["score_response"] < 0.6:
            errors.append(f"response quality: {r['score_response']:.2f} — {r.get('judge_notes', '')}")
        if errors:
            print(f"  [{r['subject'][:40]}] {', '.join(errors)}")

    # ── Write eval CSV ────────────────────────────────────────────────────────

    eval_fields = [
        "issue", "subject", "company",
        "expected_status", "predicted_status", "score_status",
        "expected_request_type", "predicted_request_type", "score_request_type",
        "expected_product_area", "predicted_product_area", "score_product_area",
        "score_hallucination", "score_completeness", "score_response", "overall",
        "predicted_response", "expected_response", "justification", "judge_notes",
    ]
    with open(EVAL_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eval_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed results written to: {EVAL_OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent against sample_support_tickets.csv")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM response quality scoring")
    parser.add_argument("--verbose", action="store_true", help="Show predicted vs expected responses")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N tickets (0=all)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild FAISS index")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
