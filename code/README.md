# Support Ticket Triage Agent

Terminal-based agent that resolves support tickets across HackerRank, Claude, and Visa using local RAG + Claude haiku.

**Requires:** Python 3.10+

---

## Architecture

```
CSV Input
   ↓
[1] Preprocessor    — normalize ticket, infer company when None
   ↓
[2] Injection check — detect prompt injection / jailbreak attempts
   ↓
[3] Escalation rules — regex patterns: fraud, outage, billing, security
   ↓
[4] FAISS Retrieval — domain-filtered top-5 docs from 774-doc corpus
   ↓
[5] Low-confidence gate — max similarity < 0.25 → escalate
   ↓
[6] Claude haiku (tool_use) — structured JSON output, temperature=0
   ↓
[7] Hard override — force_escalate enforced AFTER LLM response
   ↓
CSV Output
```

| Module | Purpose |
|---|---|
| `indexer.py` | Build/cache FAISS index from 774 corpus docs |
| `retriever.py` | Semantic search (sentence-transformers + FAISS), optional cross-encoder reranking |
| `classifier.py` | Rule-based escalation (19 patterns), injection detection (7 patterns), company inference |
| `prompts.py` | System prompt + 3 few-shot examples (reply, escalate, invalid) |
| `agent.py` | Full pipeline: safety gates → retrieval → LLM → validation |
| `main.py` | CLI entry point, CSV I/O, seeded randomness |
| `eval.py` | Evaluate against `sample_support_tickets.csv` with LLM judge scoring |
| `tests/` | 60 unit tests (classifier + indexer utilities) |

---

## Setup

```bash
# From repo root
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r code/requirements.txt

cp .env.example .env
# Edit .env → ANTHROPIC_API_KEY=sk-ant-...
```

**After installing, pin exact versions:**
```bash
pip freeze > code/requirements_lock.txt
```

---

## Run

```bash
# Full run — reads support_tickets/support_tickets.csv, writes support_tickets/output.csv
python code/main.py

# Verbose (shows company inference, retrieval score, per-ticket decision)
python code/main.py --verbose

# With cross-encoder reranking (slower, more accurate retrieval)
python code/main.py --rerank

# Force rebuild FAISS index (first run builds automatically, ~30s)
python code/main.py --rebuild-index

# Dev run — sample CSV has expected outputs for comparison
python code/main.py \
  --input support_tickets/sample_support_tickets.csv \
  --output support_tickets/sample_output.csv
```

---

## Evaluate (before final run)

Run against `sample_support_tickets.csv` (108 rows with gold labels):

```bash
# Fast eval — no API call, embedding cosine similarity vs gold, spot-check 10 justifications
python code/eval.py --no-llm-judge --semantic-sim --spot-check 10

# Full eval with LLM judge (scores hallucination + completeness)
python code/eval.py --verbose

# Classify failure types and write analysis_report.txt
python code/eval.py --no-llm-judge --analyze

# Validate output.csv integrity (56 rows, 5 cols, no empty replied)
python code/eval.py --validate-output

# Evaluate only first N tickets (dev/debug)
python code/eval.py --limit 5
```

Reports: per-field accuracy (status, request_type, product_area), per-domain breakdown (HackerRank/Claude/Visa), escalation miss alarm, optional semantic similarity and LLM judge scores. Writes `support_tickets/eval_output.csv`.

---

## Unit Tests

```bash
python -m pytest code/tests/ -v
```

60 tests covering escalation rules, injection detection, company inference, and corpus parsing utilities. No API key required.

---

## Per-Ticket Pipeline

1. **Company normalization** — `None`/blank/`"None"` → infer from keyword heuristics
2. **Injection detection** — French/English prompt injection, jailbreaks → `replied/invalid`
3. **Hard escalation rules** — 19 regex patterns (fraud, outage, billing, security, access)
4. **Semantic retrieval** — embed query, FAISS cosine search, domain-filtered, top-5 docs
5. **Low-confidence gate** — `max_similarity < 0.25` → escalate (corpus can't answer)
6. **LLM call** — Claude haiku, `temperature=0`, `tool_use` forces JSON schema compliance
7. **Hard override** — `force_escalate` enforced after LLM (LLM cannot bypass safety rules)
8. **Output validation** — enum enforcement, non-empty response for replied tickets

### Multi-request tickets
When a ticket contains multiple requests, the LLM addresses the primary intent in one pass and notes the secondary requests in the justification. If any part of the ticket triggers an escalation rule, the entire ticket is escalated.

---

## Output

`support_tickets/output.csv` — 56 rows, 8 columns:

| Column | Allowed Values |
|---|---|
| `status` | `replied` \| `escalated` |
| `product_area` | snake_case support category (from corpus breadcrumbs) |
| `response` | user-facing answer, grounded in provided corpus only |
| `justification` | routing explanation traceable to corpus |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` |

---

## Design Decisions

| Decision | Choice | Why not alternatives |
|---|---|---|
| LLM | Claude haiku | temperature=0 support, tool_use for schema, ANTHROPIC_API_KEY available |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free, offline, no API cost, sufficient for 774 docs |
| Vector store | FAISS `IndexFlatIP` | No server, exact search, deterministic, trivial at <1k docs |
| Retrieval | Domain-filtered (per company) | Prevents cross-domain noise, boosts precision |
| Escalation | Rules-first, then LLM | Deterministic safety net; LLM cannot override hard rules |

---

## Notes

- `code/.cache/` stores built FAISS index — gitignored, rebuilt on first run
- `.env` is gitignored — never commit API keys
- Do not commit `code/.cache/`, `.env`, or `.venv/`
