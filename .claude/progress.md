# Progress Tracker — HackerRank Orchestrate

**Challenge deadline:** May 2, 2026, 11:00 AM IST

---

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done

---

## Phase 1 — Setup & Planning
- [x] Read problem statement + evaluation criteria
- [x] Explored corpus structure (774 docs: HackerRank 438, Claude 322, Visa 14)
- [x] Analyzed sample_support_tickets.csv (109 rows with expected outputs)
- [x] Designed architecture (RAG + rule-based escalation + Claude haiku)
- [x] Updated CLAUDE.md with full context, dos/don'ts, commands
- [x] Created .claude/progress.md
- [ ] Onboarding agreement logged to ~/hackerrank_orchestrate/log.txt

## Phase 2 — Infrastructure
- [x] `code/requirements.txt` — pin all dependencies
- [x] `.env.example` — template with ANTHROPIC_API_KEY
- [x] `code/indexer.py` — FAISS index builder from data/ corpus
- [x] `code/retriever.py` — semantic search wrapper
- [x] `code/classifier.py` — escalation rules + company inference
- [x] `code/prompts.py` — system prompt + few-shot examples

## Phase 3 — Agent Core
- [x] `code/agent.py` — Claude API + tool_use structured output + pipeline
- [x] `code/main.py` — CLI entry point, CSV I/O, progress reporting
- [x] `code/README.md` — setup and run instructions

## Phase 4 — Testing & Tuning
- [x] Unit tests: code/tests/test_classifier.py (45 tests — 45/45 passing)
- [x] Unit tests: code/tests/test_indexer_utils.py (15 tests — 15/15 passing)
- [x] Eval framework: code/eval.py (LLM judge + per-field scoring)
- [ ] Run eval.py against sample_support_tickets.csv (needs ANTHROPIC_API_KEY + installed deps)
- [ ] Tune escalation thresholds based on eval results
- [ ] Tune prompts for response quality

## Phase 5 — Packaging
- [ ] `code/README.md` — install + run instructions
- [ ] Final run on support_tickets.csv → output.csv
- [ ] Verify output.csv: 56 rows, 5 columns, no nulls
- [ ] Zip code/ for submission
- [ ] Export log.txt from ~/hackerrank_orchestrate/log.txt

---

## Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| LLM | claude-haiku-4-5-20251001 | Fast, cheap, best instruction-following, ANTHROPIC_API_KEY available |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | Free, offline, sufficient quality for 774 docs |
| Vector store | FAISS in-memory + disk cache | No server, exact search, trivial at this scale |
| Retrieval strategy | Domain-filtered (per company) | Prevents cross-domain noise, boosts precision |
| Escalation | Rules-first, then LLM | Deterministic safety net for fraud/outage cases |
| Temperature | 0 | Determinism required by eval criteria |

---

## Issues / Blockers

_None yet._

---

## Notes

- `sample_support_tickets.csv` has ground truth — use it to validate before final run
- Visa corpus is tiny (14 files) — Visa tickets may rely more on LLM judgment + escalation
- Some tickets have `Company=None` — company inference is critical for retrieval quality
- Low similarity scores (<0.25) should trigger escalation — can't answer what corpus doesn't cover
