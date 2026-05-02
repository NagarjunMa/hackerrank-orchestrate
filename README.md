# HackerRank Orchestrate — Multi-Domain Support Triage Agent

**24-hour hackathon | May 1–2, 2026**

Terminal-based AI agent that triages real support tickets across HackerRank, Claude, and Visa. Reads a CSV of tickets, produces structured decisions — status, product area, response, justification, request type — grounded entirely in a 774-doc local corpus. No live web calls. No hallucinated policies.

---

## Agent Pattern

This agent uses a **Hybrid RAG + Rule-Based Escalation** pattern — not a pure LLM chain, not a pure rules engine. The two work together:

- **Rules gate first.** 19 escalation patterns + 7 injection detectors run before the LLM sees the ticket. These are deterministic — the model cannot override them.
- **RAG provides context.** Relevant docs are retrieved per ticket and injected into the LLM prompt. The LLM answers only from what it receives.
- **Rules gate again after.** If `force_escalate=True` was set pre-LLM, the result is overwritten post-LLM regardless of what the model returned.

This design prevents the failure mode where a well-meaning LLM tries to "help" with fraud, outages, or security issues by fabricating reassuring steps.

```
CSV Input
   ↓
[1] Preprocessor       — normalize ticket text, infer company when None/blank
   ↓
[2] Injection check    — 7 patterns: French/English prompt injection, jailbreaks → replied/invalid
   ↓
[3] Escalation rules   — 19 regex patterns: fraud, outage, billing, security, access → force_escalate=True
   ↓
[4] FAISS Retrieval    — embed query, cosine search, domain-filtered by company, top-5 docs
   ↓
[5] Low-confidence gate — max similarity < 0.25 → hint to LLM (corpus can't answer)
   ↓
[6] Claude haiku        — tool_use for schema-enforced JSON output, temperature=0
   ↓
[7] Hard override       — force_escalate enforced AFTER LLM response (LLM cannot bypass)
   ↓
[8] Output validation  — enum check, non-empty response on replied tickets
   ↓
CSV Output
```

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | `claude-haiku-4-5-20251001` | `tool_use` enforces JSON schema, `temperature=0` for determinism, fastest haiku model |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | Free, fully offline, no API cost, 384-dim, sufficient quality for 774 docs |
| Vector store | `faiss-cpu` `IndexFlatIP` | No server setup, exact cosine search, deterministic, index cached to disk |
| Retrieval strategy | Domain-filtered per company | FAISS searched only within HackerRank / Claude / Visa subdomain — prevents cross-domain noise |
| Company inference | Keyword heuristics | Tickets with `Company=None` matched against domain keyword sets before retrieval |
| Escalation | Regex rules-first, LLM-last | Deterministic safety net — 19 patterns catch fraud/outage/security before LLM involvement |
| Schema enforcement | Claude `tool_use` | Forces structured output with enum validation — no free-text parsing needed |
| Language | Python 3.10+ | Best ML tooling, clean path imports, csv/pathlib stdlib |
| Secrets | `python-dotenv` | Reads `ANTHROPIC_API_KEY` from `.env`, never hardcoded |

---

## Folder Structure

```
.
├── README.md                       ← you are here
├── AGENTS.md                       ← hackathon rules + AI tool logging config
├── problem_statement.md            ← original task spec
├── evalutation_criteria.md         ← scoring rubric
├── .env.example                    ← copy to .env, add API key
│
├── code/                           ← all agent source
│   ├── main.py                     ← entry point: reads CSV, runs pipeline, writes output
│   ├── agent.py                    ← orchestrates full pipeline per ticket
│   ├── retriever.py                ← FAISS semantic search, domain filter, confidence scoring
│   ├── classifier.py               ← escalation rules (19), injection detection (7), company inference
│   ├── indexer.py                  ← build + cache FAISS index from data/ corpus
│   ├── prompts.py                  ← system prompt + few-shot examples
│   ├── eval.py                     ← evaluate against gold labels, per-domain breakdown, failure analysis
│   ├── requirements.txt            ← pinned exact versions (==)
│   ├── .cache/                     ← FAISS index binary + metadata (gitignored, auto-built)
│   └── tests/
│       ├── test_classifier.py      ← 50+ tests: escalation rules, injection, company inference
│       └── test_indexer_utils.py   ← corpus parsing utilities
│
├── support_tickets/
│   ├── support_tickets.csv         ← input: 29 real tickets (Issue, Subject, Company)
│   ├── sample_support_tickets.csv  ← dev set: 108 tickets with gold labels
│   └── output.csv                  ← agent output: 29 rows × 8 cols (submit this)
│
└── data/
    ├── hackerrank/                 ← 438 markdown docs (screen, interviews, library, etc.)
    ├── claude/                     ← 322 markdown docs (api, billing, privacy, desktop, etc.)
    └── visa/                       ← 14 markdown docs (consumer support, card services)
```

---

## Output Schema

`support_tickets/output.csv` — one row per input ticket:

| Column | Allowed Values | How it's set |
|---|---|---|
| `status` | `replied` \| `escalated` | Rules + LLM, hard override post-LLM |
| `product_area` | snake_case category | Derived from retrieved doc breadcrumbs |
| `response` | user-facing answer | LLM, grounded in retrieved docs only |
| `justification` | routing explanation | LLM, references corpus and rule trigger |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` | LLM via tool_use schema |

---

## Setup

**Requirements:** Python 3.10+ · `ANTHROPIC_API_KEY`

```bash
git clone git@github.com:interviewstreet/hackerrank-orchestrate-may26.git
cd hackerrank-orchestrate-may26

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r code/requirements.txt

cp .env.example .env
# Open .env and set: ANTHROPIC_API_KEY=sk-ant-...
```

---

## Run

```bash
# Full run — reads support_tickets.csv, writes output.csv (~2–3 min for 29 tickets)
python code/main.py

# Verbose — shows company inference, retrieval scores, decision per ticket
python code/main.py --verbose

# Force rebuild FAISS index (auto-built on first run, cached after)
python code/main.py --rebuild-index
```

---

## Test

### Unit tests (no API key needed)

```bash
python -m pytest code/tests/ -v
```

60 tests covering escalation rule matching, injection detection, company inference, and corpus parsing. Runs fully offline.

### Eval against gold labels

`sample_support_tickets.csv` has 108 tickets with expected outputs. Use it to measure accuracy before the final run:

```bash
# Fast eval — cosine similarity vs gold, spot-check 10 worst justifications
python code/eval.py --no-llm-judge --semantic-sim --spot-check 10

# Full eval — LLM judge scores hallucination + completeness per response
python code/eval.py --verbose

# Classify failure types → writes support_tickets/analysis_report.txt
python code/eval.py --no-llm-judge --analyze

# Validate output.csv structure before submission
python code/eval.py --validate-output
```

Eval reports:
- Per-field accuracy: `status`, `request_type`, `product_area`
- Per-domain breakdown: HackerRank / Claude / Visa separately
- Escalation miss alarm — flags any ticket where expected=escalated but predicted=replied
- Optional: LLM hallucination + completeness score, semantic similarity vs gold response

---

## Escalation Rules (Hard Patterns)

These 19 patterns fire **before** the LLM and are re-enforced **after**. LLM output cannot override them:

| Category | Example triggers |
|---|---|
| Assessment fraud | score manipulation, force pass, unfair grading, answer disclosure |
| Platform outage | site/service/platform is down, nothing works, all submissions failing |
| Billing dispute | demand refund now, chargeback, force money back |
| Security breach | account hacked, stolen credentials, unauthorized access |
| Access/permissions | lost workspace seat, admin removed my access |
| Injection detection | prompt injection attempts, jailbreak patterns |

---

## Key Design Decisions

**Why rules before LLM?**
LLMs trained on helpful data will try to assist with everything. A regex that matches "force me to pass" fires in <1ms and requires no API call. Deterministic, auditable, zero cost.

**Why domain-filtered retrieval?**
A HackerRank ticket searched against the full 774-doc corpus would surface Claude billing docs as false positives. Filtering to the company domain reduces noise and improves top-5 doc relevance significantly.

**Why `tool_use` instead of prompt-asking for JSON?**
Claude's `tool_use` enforces the output schema at the API level — enum values, required fields, no extra keys. Eliminates an entire class of parsing bugs.

**Why `temperature=0`?**
Reproducibility. Same ticket in → same ticket out. Easier to debug, easier to evaluate, safer for a safety-critical triage system.

**Why cache the FAISS index?**
Building the index (encoding 774 docs) takes ~30 seconds. Caching to `code/.cache/` means every run after the first starts in under 2 seconds.
