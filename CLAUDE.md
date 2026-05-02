@AGENTS.md

---

# CLAUDE.md — Agent Context for HackerRank Orchestrate

## Problem Statement

Build a **terminal-based AI agent** that triages real support tickets across three product ecosystems:
- **HackerRank** — hiring/screening platform support
- **Claude** — Anthropic's AI assistant support
- **Visa** — consumer card support

### Input
`support_tickets/support_tickets.csv` — 56 tickets with fields: `Issue`, `Subject`, `Company`

### Required Output
`support_tickets/output.csv` — 5 fields per ticket:

| Field | Allowed Values |
|---|---|
| `status` | `replied` \| `escalated` |
| `product_area` | support category/domain (derived from corpus) |
| `response` | user-facing answer, grounded in corpus only |
| `justification` | concise explanation of routing decision |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` |

### Corpus
774 markdown files in `data/`:
- `data/hackerrank/` — 438 files (screen, interviews, library, settings, integrations, etc.)
- `data/claude/` — 322 files (api, privacy, billing, desktop, mobile, etc.)
- `data/visa/` — 14 files (consumer support, card services)

---

## Approach

**Hybrid RAG + Rule-based escalation pipeline:**

```
CSV Input → Preprocessor → Retriever → Escalation Check → LLM → CSV Output
```

1. **Preprocessor** — normalize ticket text, infer `Company` when `None`
2. **Retriever** — embed query with sentence-transformers, FAISS cosine search, return top-5 docs (domain-filtered)
3. **Escalation pre-check** — regex rules for fraud/outage/billing/security (deterministic, no LLM)
4. **LLM (Claude haiku)** — structured output via tool_use, corpus-grounded response
5. **CSV Writer** — write all 5 fields to `output.csv`

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | `claude-haiku-4-5-20251001` | Best instruction-following, native tool_use JSON, ANTHROPIC_API_KEY available |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | Free, offline, fast, 384-dim, sufficient quality for 774 docs |
| Vector Index | `faiss-cpu` `IndexFlatIP` | No server, exact search fine at <1k docs, pure Python |
| Language | Python 3.10+ | Clear winner for ML tooling |
| Env management | `python-dotenv` | Read secrets from `.env` |

---

## Module Layout

```
code/
├── main.py          # Entry point — orchestrates full pipeline
├── indexer.py       # Build + cache FAISS index from data/
├── retriever.py     # Semantic search wrapper
├── classifier.py    # Escalation rules + company inference
├── agent.py         # Claude API call + structured output parsing
├── prompts.py       # System prompt + few-shot examples
├── requirements.txt # All dependencies
└── README.md        # Install + run instructions
```

Index cache: `code/.cache/index.faiss` + `code/.cache/metadata.json` (auto-built on first run)

---

## DOs

- **DO** read secrets from env vars only (`ANTHROPIC_API_KEY`)
- **DO** answer only from retrieved corpus — quote or paraphrase docs
- **DO** escalate: fraud, score manipulation, billing disputes, platform outages, security breaches
- **DO** escalate when retrieval confidence is low (similarity score < 0.25) — can't answer from corpus
- **DO** set `temperature=0` on all Claude API calls for determinism
- **DO** cache the FAISS index to disk — don't rebuild every run
- **DO** use `Company` field to filter corpus before retrieval (domain-aware search)
- **DO** infer company from ticket content when `Company=None`
- **DO** derive `product_area` from breadcrumbs in retrieved docs, not hardcoded
- **DO** include 3 few-shot examples in system prompt (procedural reply, escalation, invalid)
- **DO** seed random: `torch.manual_seed(42)`, `numpy.random.seed(42)`
- **DO** write `output.csv` with all 5 columns for all 56 rows

## DON'Ts

- **DON'T** hallucinate policies, steps, phone numbers, or URLs not in corpus
- **DON'T** make live web calls for answers (corpus-only)
- **DON'T** hardcode API keys
- **DON'T** stuff all 774 docs into one context window
- **DON'T** try to "help" with fraud or score manipulation requests — always escalate
- **DON'T** use keyword/exact match as primary retrieval — semantic gaps will kill accuracy
- **DON'T** return empty `response` for `replied` tickets
- **DON'T** commit `.env`, `code/.cache/`, or virtualenv files

---

## Escalation Patterns (Hard Rules)

These always force `status=escalated` regardless of retrieved docs:

```python
ESCALATION_PATTERNS = [
    r"score.*manipulat|force.*pass|unfair.*grad|change.*result",   # assessment fraud
    r"platform.*down|site.*down|everything.*broken|nothing.*work", # critical outage
    r"force.*refund|chargeback|payment.*dispute",                  # billing dispute
    r"account.*hack|stolen.*cred|compromis|unauthorized.*access",  # security breach
    r"remove.*seat|workspace.*access|permission.*denied",          # auth/permissions
]
```

Also escalate if `max_similarity_score < 0.25` (corpus can't answer the question).

---

## Company Inference (when Company=None)

```python
COMPANY_KEYWORDS = {
    "claude": ["claude", "anthropic", "api key", "api_key", "console", "claude.ai"],
    "hackerrank": ["hackerrank", "test", "assessment", "coding challenge", "interview", "candidate", "recruiter"],
    "visa": ["visa", "card", "merchant", "payment", "transaction", "chargeback", "atm"],
}
```

If no match → search all domains.

---

## Bash Commands

### Setup
```bash
# Clone (if needed)
git clone git@github.com:interviewstreet/hackerrank-orchestrate-may26.git
cd hackerrank-orchestrate-may26

# Create virtualenv
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r code/requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...
```

### Run Agent
```bash
# Full pipeline — reads support_tickets.csv, writes output.csv
python code/main.py

# Run on sample tickets only (for dev/testing)
python code/main.py --input support_tickets/sample_support_tickets.csv --output support_tickets/sample_output.csv

# Force rebuild FAISS index
python code/main.py --rebuild-index

# Verbose mode
python code/main.py --verbose
```

### Validate Output
```bash
# Check output row count
wc -l support_tickets/output.csv

# Check all 5 columns present
head -3 support_tickets/output.csv

# Spot check specific tickets
python -c "import csv; rows=list(csv.DictReader(open('support_tickets/output.csv'))); print(rows[0])"
```

### Debug / Dev
```bash
# Test retriever on a single query
python -c "from code.retriever import Retriever; r=Retriever(); print(r.retrieve('how to delete account', company='hackerrank'))"

# Test indexer
python -c "from code.indexer import build_index; build_index(force=True)"

# Count corpus files
find data/ -name "*.md" | wc -l
find data/hackerrank -name "*.md" | wc -l
find data/claude -name "*.md" | wc -l
find data/visa -name "*.md" | wc -l

# Check .env loaded
python -c "import dotenv; dotenv.load_dotenv(); import os; print('KEY SET:', bool(os.getenv('ANTHROPIC_API_KEY')))"
```

### Git
```bash
git status
git diff
git add code/
git commit -m "feat: add support ticket triage agent"
```

---

## Key File Paths

| Purpose | Path |
|---|---|
| Eval tickets (input) | `support_tickets/support_tickets.csv` |
| Sample tickets (dev) | `support_tickets/sample_support_tickets.csv` |
| Agent output | `support_tickets/output.csv` |
| Corpus root | `data/` |
| HackerRank corpus | `data/hackerrank/` |
| Claude corpus | `data/claude/` |
| Visa corpus | `data/visa/` |
| Entry point | `code/main.py` |
| Index cache | `code/.cache/` |
| Env template | `.env.example` |
| Env secrets | `.env` (gitignored) |
| Progress tracker | `.claude/progress.md` |
| Log file | `~/hackerrank_orchestrate/log.txt` |

---

## Evaluation Dimensions

1. **Agent Design** (code architecture, separation of concerns, RAG quality, escalation logic)
2. **AI Judge Interview** (defend design decisions, trade-offs, failure modes)
3. **Output CSV Accuracy** (all 5 columns scored per row)
4. **AI Fluency** (chat transcript shows user steering decisions, critique, verification)

**Zero tolerance for:** hallucinated policies, fabricated steps, guessing on high-risk tickets.
