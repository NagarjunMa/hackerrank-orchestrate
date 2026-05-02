# Evaluation Criteria Checklist — Full Audit

Generated: 2026-05-01
Status legend: ✅ DONE | ⚠️ PARTIAL | ❌ MISSING / BROKEN

---

## DIMENSION 1 — Agent Design (code/ directory)

### Architecture & Approach
- ✅ Clear separation of concerns: indexer → retriever → classifier → agent → main
- ✅ RAG technique chosen and implemented (FAISS + sentence-transformers)
- ✅ Structured output via Claude tool_use (not free-form text parsing)
- ✅ Company inference when Company=None
- ✅ Domain-filtered retrieval (per-company corpus subset)
- ✅ Three-gate safety: injection → escalation rules → low-confidence
- ⚠️ **Multi-request tickets not explicitly handled** — problem says "a row may contain multiple requests". Current: LLM handles in one pass, no splitting. Should document this design choice explicitly.

### Use of Provided Corpus
- ✅ Retrieves from data/ only (no web calls)
- ✅ System prompt: "only from provided documentation"
- ✅ Source URLs from corpus included in docs context
- ✅ Breadcrumbs used for product_area derivation
- ⚠️ **Corpus content truncated at 800 chars in _format_docs()** — long docs may miss critical info. Consider increasing to 1200-1500 chars.
- ⚠️ **Title-only search** for embed text — could weight title more explicitly

### Escalation Logic
- ✅ Rule-based: fraud, outage, billing dispute, security breach, access restoration
- ✅ Injection detection: French, English, jailbreak patterns
- ✅ Low-confidence gate: max_score < 0.25 → escalate
- ✅ Hard override: force_escalate enforced AFTER LLM call
- ✅ Fallback result: always escalates on LLM failure
- ⚠️ **LOW_CONFIDENCE_THRESHOLD = 0.25 untested** — may be too strict (escapes legitimate questions) or too loose. Needs eval run to calibrate.
- ⚠️ **Subscription pause (ticket 14)** — no hard rule fires; LLM handles. Correct behavior but untested.

### Determinism & Reproducibility
- ✅ temperature=0 on all Claude API calls
- ✅ numpy.random.seed(42) in main.py
- ✅ torch.manual_seed(42) in main.py
- ✅ FAISS IndexFlatIP is deterministic (exact search)
- ✅ code/README.md exists with install + run instructions
- ❌ **requirements.txt uses `>=` not pinned versions** — eval criteria explicitly requires "pinned dependencies". MUST fix.
  - faiss-cpu==1.13.2 (confirmed installed)
  - numpy==2.4.4 (confirmed installed)
  - python-dotenv==1.2.2 (confirmed installed)
  - PyYAML==6.0.3 (confirmed installed)
  - anthropic / sentence-transformers: pin AFTER `pip install -r code/requirements.txt`
- ❌ **anthropic NOT installed in .venv** — needs `pip install anthropic`
- ❌ **sentence-transformers NOT installed in .venv** — needs `pip install sentence-transformers`

### Engineering Hygiene
- ✅ Secrets from env vars only (ANTHROPIC_API_KEY)
- ✅ No hardcoded keys anywhere in codebase
- ✅ .env gitignored
- ✅ .venv gitignored
- ✅ Readable code with docstrings
- ✅ Sensible module structure
- ❌ **code/.cache/ NOT in .gitignore** — FAISS index binary (large file) will be committed when generated. Must add to .gitignore.
- ⚠️ **pytest not in requirements.txt** — tests exist but can't be run reproducibly without pytest
- ⚠️ **Python version not specified in README** — requires 3.10+ (str|None syntax)

---

## DIMENSION 2 — AI Judge Interview (30 min, camera on)

### Depth of Understanding
- ✅ Can explain: RAG chosen over keyword search, FAISS over vector DB, haiku over GPT-4, sentence-transformers over OpenAI embeddings
- ⚠️ **No written failure modes doc** — should prepare:
  - Where agent breaks: Visa corpus is tiny (14 docs) → many Visa tickets escalate due to low confidence
  - Where agent breaks: ambiguous company=None tickets may mismatch domain
  - Where agent breaks: very long tickets may get truncated context
  - Where agent breaks: cross-domain tickets (Claude + HackerRank in one ticket)

### Trade-off Awareness
- ✅ CLAUDE.md documents tech stack decisions
- ⚠️ **Need to document what was rejected and WHY**:
  - ChromaDB rejected: server overhead unnecessary for 774 docs
  - GPT-4o rejected: Claude available, better instruction-following, lower cost
  - BM25 rejected: pure keyword match misses semantic queries; hybrid considered but FAISS sufficient
  - Fine-tuning rejected: no labeled training data; overkill for 56 tickets
  - Stuffing all docs rejected: 774×500 tokens = 387k tokens, expensive/noisy

### Failure-Mode Reasoning
- ⚠️ **Prepare answers to**:
  - "What happens if Claude API is down?" → Fallback result: escalate all tickets
  - "What if sentence-transformers fails to load?" → Immediate crash at startup; could add health check
  - "What if a ticket is in a language other than English?" → Injection detection (French) works, but French support ticket would get classified incorrectly
  - "What if the corpus is outdated?" → Agent will give stale answers; recommend corpus versioning

### Honesty About AI Assistance
- ✅ log.txt exists showing AI-assisted development
- ⚠️ **log.txt entries need to show user steering** — entries currently show agent doing work; need entries where user critiqued/corrected AI decisions

---

## DIMENSION 3 — Output CSV (support_tickets/output.csv)

### Column Presence & Format
- ✅ All 8 columns present in header: issue, subject, company, response, product_area, status, request_type, justification
- ❌ **output.csv is EMPTY** — only header row, no data. MUST run `python code/main.py` before submission.

### Status Accuracy
- ⚠️ **Not yet verified** — need eval run
- Expected escalations (from support_tickets.csv analysis):
  - Row 1: Claude workspace access restored → ESCALATE (lost access)
  - Row 2: Score manipulation → ESCALATE (fraud rule)
  - Row 3: Force Visa refund + ban seller → ESCALATE (forced refund rule)
  - Row 4: Mock interview refund asap → ESCALATE (refund rule)
  - Row 5: Payment with order ID → ESCALATE (billing + PII)
  - Row 6: Infosec forms for HackerRank → likely ESCALATE (enterprise sales, not support)
  - Row 8: Submissions not working across all challenges → ESCALATE (systemic failure)
  - Row 15: Claude stopped working completely → ESCALATE (outage rule)
  - Row 16: Identity theft → ESCALATE (identity theft rule)
  - Row 17: Resume Builder is Down → ESCALATE (is down rule)
  - Row 19: "How do I dispute a charge" → REPLY (general question, answerable from Visa corpus)
  - Row 24: "Give me code to delete all files" → REPLY / invalid (out of scope, not injection)
  - Row 25: French prompt injection → REPLY / invalid (injection detection)

### Product Area Accuracy
- ⚠️ **Not yet verified**
- Expected from sample: screen, community, privacy, travel_support, general_support, conversation_management
- Key: must match known product_area values for each company
- Risk: LLM may invent product areas not in corpus

### Response Quality
- ✅ System prompt: no hallucination instruction
- ✅ Retrieved docs in context for grounding
- ✅ _validate_result() enforces non-empty response for replied tickets
- ⚠️ **Not yet run** — need actual output to verify

### Justification Quality
- ✅ tool_use schema requires justification
- ⚠️ **Not yet run** — may be generic; should reference specific documents

---

## DIMENSION 4 — AI Fluency (log.txt)

### log.txt Location
- ✅ ~/hackerrank_orchestrate/log.txt exists
- ✅ Onboarding entry recorded
- ✅ Session start entry recorded
- ✅ Per-turn entries for major decisions

### Evidence of User Steering
- ⚠️ **log.txt needs entries showing**:
  - User questioning AI design choices
  - User catching bugs (e.g., "force_escalate not enforced — I caught this")
  - User deciding architecture (e.g., "I chose FAISS over ChromaDB because...")
  - User verifying outputs critically

---

## COMPLETE TODO LIST (ordered by priority)

### 🔴 CRITICAL — Must fix before submission

| # | Item | Action |
|---|------|--------|
| 1 | **output.csv empty** | Install all deps, set ANTHROPIC_API_KEY, run `python code/main.py` |
| 2 | **anthropic not installed** | `pip install anthropic` in venv |
| 3 | **sentence-transformers not installed** | `pip install sentence-transformers` |
| 4 | **requirements.txt not pinned** | Run `pip freeze` after install, update requirements.txt with `==` versions |
| 5 | **code/.cache/ not gitignored** | Add `code/.cache/` to .gitignore |

### 🟠 HIGH — Directly affects eval score

| # | Item | Action |
|---|------|--------|
| 6 | **Run eval.py** | `python code/eval.py --verbose` → check per-field accuracy |
| 7 | **Verify status decisions** | Check escalate/reply matches expected for all 56 tickets |
| 8 | **Verify product_area values** | Ensure LLM uses corpus-derived areas not invented ones |
| 9 | **Docs truncated at 800 chars** | Increase `_format_docs()` content limit to 1200+ chars |
| 10 | **LOW_CONFIDENCE_THRESHOLD calibration** | Run eval, check if 0.25 is right threshold |

### 🟡 MEDIUM — Affects design score and interview

| # | Item | Action |
|---|------|--------|
| 11 | **Add pytest to requirements.txt** | `pytest>=8.0.0` |
| 12 | **Add Python version to README** | "Requires Python 3.10+" |
| 13 | **Document multi-request handling** | Add to README: "LLM handles multiple requests in one pass, prioritizes primary intent" |
| 14 | **Document failure modes** | Add to CLAUDE.md or .claude/interview_prep.md |
| 15 | **Document trade-off decisions** | Expand CLAUDE.md with rejected alternatives |
| 16 | **code/README.md: add eval.py section** | Show how to run eval + interpret results |

### 🟢 LOW — Quality improvements

| # | Item | Action |
|---|------|--------|
| 17 | **--rerank flag not documented in README** | Add cross-encoder usage to code/README.md |
| 18 | **log.txt user-steering entries** | Add entries showing user critique of AI output |
| 19 | **Verify no .cache/ committed** | `git status` after first index build |
| 20 | **Final output.csv sanity check** | Verify 56 rows, no empty responses, valid enum values |
