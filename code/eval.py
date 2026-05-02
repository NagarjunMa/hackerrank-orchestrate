"""
eval.py — Evaluate the agent against sample_support_tickets.csv (which has ground truth).

Scoring:
  status       — exact match (replied / escalated)
  request_type — exact match (product_issue / feature_request / bug / invalid)
  product_area — fuzzy match (normalize case + underscores, check substring)
  response     — LLM judge: hallucination + completeness (0.0 – 1.0)
  overall      — average of all four field scores

Usage:
    python code/eval.py                          # full eval with LLM judge
    python code/eval.py --no-llm-judge          # skip LLM judge (faster, offline)
    python code/eval.py --verbose               # show predicted vs expected responses
    python code/eval.py --limit 5              # evaluate only first N tickets
    python code/eval.py --analyze              # classify failure types + write analysis report
    python code/eval.py --semantic-sim          # cosine similarity of response vs gold (fast, no API)
    python code/eval.py --spot-check 10        # print 10 lowest-scoring justifications
    python code/eval.py --validate-output       # check output.csv integrity (56 rows, 5 cols, etc.)
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
ANALYSIS_REPORT = REPO_ROOT / "support_tickets" / "analysis_report.txt"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"
EXPECTED_OUTPUT_ROWS = 56
REQUIRED_OUTPUT_COLS = {"status", "product_area", "response", "justification", "request_type"}
VALID_STATUSES = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

# ── Output CSV integrity check ───────────────────────────────────────────────

def validate_output_csv(path: Path) -> list[str]:
    """Check output.csv for completeness and validity. Returns list of error strings."""
    errors = []

    if not path.exists():
        return [f"File not found: {path}"]

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return ["File is empty or has no data rows"]

    actual_cols = set(rows[0].keys())
    missing_cols = REQUIRED_OUTPUT_COLS - actual_cols
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")

    row_count = len(rows)
    if row_count != EXPECTED_OUTPUT_ROWS:
        errors.append(f"Row count: {row_count} (expected {EXPECTED_OUTPUT_ROWS})")

    for i, row in enumerate(rows, 1):
        status = (row.get("status") or "").strip().lower()
        response = (row.get("response") or "").strip()
        request_type = (row.get("request_type") or "").strip().lower()

        if status not in VALID_STATUSES:
            errors.append(f"Row {i}: invalid status={status!r}")
        if request_type and request_type not in VALID_REQUEST_TYPES:
            errors.append(f"Row {i}: invalid request_type={request_type!r}")
        if status == "replied" and not response:
            errors.append(f"Row {i}: empty response for replied ticket (subject={row.get('subject', '')[:40]!r})")

    return errors


# ── Failure classification ───────────────────────────────────────────────────

CONTACT_PATTERN = re.compile(
    r'(@|\+1|1[\s\-]?800|http[s]?://|\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b|\b\d{10,}\b)'
)


def classify_failure(
    s_status: float,
    s_area: float,
    s_hallucination: float,
    s_completeness: float,
    pred_response: str,
    max_score: float,
) -> list[str]:
    """Classify failure modes for a ticket result. Returns list of failure type strings."""
    failures = []
    if s_status < 1.0:
        failures.append("WRONG_STATUS")
    if s_area < 0.6:
        failures.append("WRONG_AREA")
    if s_hallucination >= 0 and s_hallucination < 0.5:
        if CONTACT_PATTERN.search(pred_response):
            failures.append("FABRICATED_CONTACT")
        else:
            failures.append("FABRICATED_STEPS")
    if s_hallucination >= 0.7 and s_completeness >= 0 and s_completeness < 0.5:
        failures.append("INCOMPLETE")
    if max_score < 0.35:
        failures.append("CORPUS_GAP")
    return failures


def generate_analysis_report(results: list[dict]) -> str:
    """Group failures by type, print counts and fix hints. Returns report string."""
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    perfect = []

    for r in results:
        failures = r.get("failure_types", [])
        if not failures:
            perfect.append(r)
        for f in failures:
            by_type[f].append(r)

    lines = []
    lines.append("=" * 70)
    lines.append("FAILURE ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total tickets evaluated: {len(results)}")
    lines.append(f"Perfect (no failures):   {len(perfect)}")
    lines.append(f"Tickets with failures:   {len(results) - len(perfect)}")
    lines.append("")

    FIX_HINTS = {
        "WRONG_STATUS": (
            "Check classifier.py escalation rules for false positives/negatives. "
            "Review system prompt INFORMATION vs ACTION distinction."
        ),
        "WRONG_AREA": (
            "Retriever returning wrong doc domain. Check company inference + "
            "domain-filtered FAISS search. Improve breadcrumb extraction."
        ),
        "FABRICATED_CONTACT": (
            "LLM inventing phone numbers / emails / URLs. "
            "Relevant doc may be truncated (MAX_DISPLAY_CHARS=5000 too small). "
            "Add anti-hallucination prompt rule for contact details."
        ),
        "FABRICATED_STEPS": (
            "LLM adding UI steps not in docs. "
            "Prompt fix: 'Never describe UI steps unless VERBATIM from docs.' "
            "Check retrieved doc relevance — wrong doc = LLM fills gaps from training."
        ),
        "INCOMPLETE": (
            "Right direction, missing key details. "
            "Top doc may be partially relevant — increase top_k or fix retrieval query."
        ),
        "CORPUS_GAP": (
            "Retrieval score < 0.35 — corpus doesn't cover this topic. "
            "Verify data/ has relevant doc. If missing, add to corpus or accept escalation."
        ),
    }

    for failure_type in ["WRONG_STATUS", "FABRICATED_CONTACT", "FABRICATED_STEPS",
                         "WRONG_AREA", "INCOMPLETE", "CORPUS_GAP"]:
        tickets = by_type.get(failure_type, [])
        if not tickets:
            continue
        lines.append(f"── {failure_type} ({len(tickets)} tickets) " + "─" * max(0, 50 - len(failure_type)))
        lines.append(f"   Fix hint: {FIX_HINTS.get(failure_type, 'No hint available.')}")
        lines.append("")
        for r in tickets:
            subj = (r["subject"] or "(no subject)")[:50]
            lines.append(f"   [{subj}]")
            lines.append(f"     Issue: {r['issue'][:80]}")
            if failure_type == "WRONG_STATUS":
                lines.append(f"     Expected: {r['expected_status']}  Got: {r['predicted_status']}")
            elif failure_type == "WRONG_AREA":
                lines.append(f"     Expected: {r['expected_product_area']}  Got: {r['predicted_product_area']}")
            elif failure_type in ("FABRICATED_CONTACT", "FABRICATED_STEPS"):
                lines.append(f"     Hallucination score: {r['score_hallucination']:.2f}")
                lines.append(f"     Judge notes: {r.get('judge_notes', '')[:120]}")
                lines.append(f"     Predicted: {r['predicted_response'][:150]}")
            elif failure_type == "INCOMPLETE":
                lines.append(f"     Completeness: {r['score_completeness']:.2f}  Judge notes: {r.get('judge_notes', '')[:100]}")
            elif failure_type == "CORPUS_GAP":
                lines.append(f"     Max retrieval score: {r.get('max_retrieval_score', -1):.3f}")
                lines.append(f"     Top doc: {r.get('top_doc', 'unknown')[:60]}")
            lines.append("")
        lines.append("")

    return "\n".join(lines)

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
    field_scores = {"status": [], "request_type": [], "product_area": [], "response": [], "sem_sim": []}

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

        # Always retrieve docs when judge or analyze is active
        company_key = company.lower() if company and company.lower() not in ("none", "") else None
        retrieved = []
        max_retrieval_score = 0.0
        top_doc = ""
        if judge_client or args.analyze:
            retrieved = agent.retriever.retrieve(issue + " " + subject, company=company_key, top_k=5)
            max_retrieval_score = agent.retriever.max_score(retrieved)
            top_doc = retrieved[0]["title"] if retrieved else ""

        if judge_client:
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

        # Semantic similarity (embedding cosine) — reuses loaded retriever model
        sem_sim = -1.0
        if args.semantic_sim and pred_response and expected_response:
            vecs = agent.retriever.model.encode(
                [pred_response, expected_response],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            sem_sim = float(np.dot(vecs[0], vecs[1]))

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
        if sem_sim >= 0:
            field_scores["sem_sim"].append(sem_sim)

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
        if sem_sim >= 0:
            print(f"  sem_sim: {fmt_score(sem_sim)}")
        print(f"  overall: {fmt_score(overall)}")

        if args.verbose:
            print(f"  predicted response: {pred_response[:200]}")
            print(f"  expected  response: {expected_response[:200]}")
            print(f"  justification: {pred_justification[:150]}")

        failure_types = classify_failure(
            s_status, s_product_area, s_hallucination, s_completeness,
            pred_response, max_retrieval_score,
        )

        if args.analyze and failure_types:
            print(f"  FAILURES: {', '.join(failure_types)}")

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
                "failure_types": failure_types,
                "max_retrieval_score": max_retrieval_score,
                "top_doc": top_doc,
                "sem_sim": sem_sim,
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
    agg_sim = sum(field_scores["sem_sim"]) / len(field_scores["sem_sim"]) if field_scores["sem_sim"] else -1

    print(f"  status       accuracy: {fmt_score(agg_status)}  ({sum(1 for s in field_scores['status'] if s == 1.0)}/{len(field_scores['status'])} exact)")
    print(f"  request_type accuracy: {fmt_score(agg_rt)}  ({sum(1 for s in field_scores['request_type'] if s == 1.0)}/{len(field_scores['request_type'])} exact)")
    print(f"  product_area accuracy: {fmt_score(agg_pa)}")
    if agg_resp >= 0:
        print(f"  response quality:      {fmt_score(agg_resp)}")
    if agg_sim >= 0:
        print(f"  response sem_sim:      {fmt_score(agg_sim)}")

    scored_agg = [agg_status, agg_rt, agg_pa]
    if agg_resp >= 0:
        scored_agg.append(agg_resp)
    overall_agg = sum(scored_agg) / len(scored_agg)
    print(f"\n  OVERALL SCORE: {fmt_score(overall_agg)}")

    # ── Per-domain accuracy breakdown ─────────────────────────────────────────

    from collections import defaultdict
    domain_buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        domain = (r["company"] or "unknown").lower().strip()
        if domain in ("none", ""):
            domain = "unknown"
        domain_buckets[domain].append(r)

    print(f"\n{'─' * 70}")
    print(f"  {'Domain':<14} {'N':>4}  {'status':>8}  {'req_type':>9}  {'prod_area':>10}  {'sem_sim':>8}  {'response':>9}")
    print(f"{'─' * 70}")
    for domain in sorted(domain_buckets):
        bucket = domain_buckets[domain]
        n = len(bucket)
        d_status = sum(r["score_status"] for r in bucket) / n
        d_rt = sum(r["score_request_type"] for r in bucket) / n
        d_pa = sum(r["score_product_area"] for r in bucket) / n
        d_sim_vals = [r["sem_sim"] for r in bucket if r.get("sem_sim", -1) >= 0]
        d_sim_str = f"{sum(d_sim_vals)/len(d_sim_vals):.2f}" if d_sim_vals else "  n/a"
        d_resp_vals = [r["score_response"] for r in bucket if r.get("score_response", -1) >= 0]
        d_resp_str = fmt_score(sum(d_resp_vals)/len(d_resp_vals)) if d_resp_vals else "  n/a"
        print(f"  {domain:<14} {n:>4}  {fmt_score(d_status):>8}  {fmt_score(d_rt):>9}  {fmt_score(d_pa):>10}  {d_sim_str:>8}  {d_resp_str:>9}")
    print(f"{'─' * 70}")

    # ── Escalation miss alarm ─────────────────────────────────────────────────

    misses = [r for r in results if r["expected_status"] == "escalated" and r["predicted_status"] == "replied"]
    if misses:
        print(f"\n\033[31m⚠️  ESCALATION MISSES ({len(misses)}) — predicted=replied when expected=escalated\033[0m")
        for j, r in enumerate(misses, 1):
            print(f"  [{j}] Subject: {r['subject'][:50] or '(no subject)'}")
            print(f"      Issue: {r['issue'][:80]}")
            print(f"      Pred response: {r['predicted_response'][:100]}")
    else:
        print(f"\n\033[32m✓  No escalation misses — all escalated tickets correctly routed\033[0m")

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

    # ── Justification spot-check ──────────────────────────────────────────────

    if args.spot_check > 0:
        bottom_n = sorted(results, key=lambda r: r["overall"])[:args.spot_check]
        print(f"\n{'─' * 70}")
        print(f"JUSTIFICATION SPOT-CHECK — bottom {len(bottom_n)} by overall score")
        print(f"{'─' * 70}")
        for j, r in enumerate(bottom_n, 1):
            status_match = "✓" if r["score_status"] == 1.0 else "✗"
            print(f"\n[{j}] Score={r['overall']:.2f} | Subject: {r['subject'][:50] or '(no subject)'}")
            print(f"    Status: {r['predicted_status']} {status_match} (exp: {r['expected_status']})")
            print(f"    Issue: {r['issue'][:100]}")
            print(f"    Justification: {r['justification']}")

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

    if args.analyze:
        report = generate_analysis_report(results)
        print("\n" + report)
        ANALYSIS_REPORT.parent.mkdir(parents=True, exist_ok=True)
        ANALYSIS_REPORT.write_text(report, encoding="utf-8")
        print(f"Analysis report written to: {ANALYSIS_REPORT}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent against sample_support_tickets.csv")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM response quality scoring")
    parser.add_argument("--verbose", action="store_true", help="Show predicted vs expected responses")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N tickets (0=all)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild FAISS index")
    parser.add_argument("--analyze", action="store_true", help="Classify failure types and write analysis_report.txt")
    parser.add_argument("--semantic-sim", action="store_true", help="Score response using embedding cosine similarity vs gold")
    parser.add_argument("--spot-check", type=int, default=0, metavar="N", help="Print N lowest-scoring justifications for manual review")
    parser.add_argument("--validate-output", action="store_true", help="Check output.csv integrity (row count, cols, no empty replied) then exit")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Path to output.csv for --validate-output")
    args = parser.parse_args()

    if args.validate_output:
        path = Path(args.output_path)
        print(f"Validating: {path}")
        errors = validate_output_csv(path)
        if errors:
            print(f"\033[31mFAIL — {len(errors)} issue(s):\033[0m")
            for e in errors:
                print(f"  • {e}")
            sys.exit(1)
        else:
            print(f"\033[32mPASS — {EXPECTED_OUTPUT_ROWS} rows, required columns present, no empty replied responses\033[0m")
        return

    run_eval(args)


if __name__ == "__main__":
    main()
