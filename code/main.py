"""
main.py — Entry point for the HackerRank Orchestrate support ticket triage agent.

Usage:
    python code/main.py
    python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
    python code/main.py --rebuild-index --verbose
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Seed for determinism before any imports that use random state
np.random.seed(42)

try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass

# Add code/ directory to sys.path so sibling modules resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

from agent import Agent  # noqa: E402 (after sys.path manipulation)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = REPO_ROOT / "support_tickets" / "support_tickets.csv"
DEFAULT_OUTPUT = REPO_ROOT / "support_tickets" / "output.csv"

# Output column order must match output.csv header
OUTPUT_FIELDS = ["issue", "subject", "company", "response", "product_area", "status", "request_type", "justification"]


def load_tickets(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_output(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate — support ticket triage agent")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to input CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output CSV")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild FAISS index")
    parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranking (slower but more accurate)")
    parser.add_argument("--verbose", action="store_true", help="Print debug info per ticket")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    tickets = load_tickets(input_path)
    if not tickets:
        print("Error: no tickets found in input CSV")
        sys.exit(1)

    print(f"Loaded {len(tickets)} tickets from {input_path.name}")

    # Initialize agent (builds/loads FAISS index once)
    agent = Agent(rebuild_index=args.rebuild_index, use_reranker=args.rerank, verbose=args.verbose)

    results: list[dict] = []
    for i, ticket in enumerate(tickets, 1):
        subject_preview = (ticket.get("Subject") or "")[:50]
        print(f"[{i:>2}/{len(tickets)}] {subject_preview or '(no subject)'}")

        prediction = agent.process(ticket)

        row = {
            "issue": ticket.get("Issue", ""),
            "subject": ticket.get("Subject", ""),
            "company": ticket.get("Company", ""),
            "response": prediction.get("response", ""),
            "product_area": prediction.get("product_area", ""),
            "status": prediction.get("status", "escalated"),
            "request_type": prediction.get("request_type", "product_issue"),
            "justification": prediction.get("justification", ""),
        }
        results.append(row)

        if args.verbose:
            print(
                f"       status={row['status']} | "
                f"type={row['request_type']} | "
                f"area={row['product_area']}"
            )

    write_output(results, output_path)
    print(f"\nDone. {len(results)} rows written to {output_path}")

    # Quick sanity check
    replied = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    print(f"  replied={replied}  escalated={escalated}")


if __name__ == "__main__":
    main()
