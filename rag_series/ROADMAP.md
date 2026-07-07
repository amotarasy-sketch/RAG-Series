# Roadmap

This roadmap is written in a GitHub issue style so each item can be copied into a real issue, assigned, labeled, and tracked by milestone.

## Status legend

- [ ] Not started
- [~] In progress
- [x] Done

## Milestone 1 — Developer Experience & CI

### Issue 1: Add GitHub Actions CI for tests and linting

**Type:** `chore`  
**Labels:** `ci`, `testing`, `developer-experience`  
**Priority:** High

**Problem**  
The project already has pytest and ruff configuration, but contributors need an automated quality gate for pull requests.

**Scope**

- [ ] Add a GitHub Actions workflow for Python 3.10 and 3.11.
- [ ] Install the project with development dependencies.
- [ ] Run `ruff check .`.
- [ ] Run `ruff format --check .`.
- [ ] Run `pytest`.
- [ ] Cache pip dependencies where practical.

**Acceptance criteria**

- CI runs automatically on pull requests and pushes to the main branch.
- A failing test or lint error blocks the workflow.
- Workflow status is visible in the repository actions tab.

---

### Issue 2: Add pre-commit hooks

**Type:** `chore`  
**Labels:** `developer-experience`, `quality`  
**Priority:** Medium

**Problem**  
Formatting and linting are currently manual steps.

**Scope**

- [ ] Add `.pre-commit-config.yaml`.
- [ ] Include ruff lint and format hooks.
- [ ] Document installation and usage in `README.md`.

**Acceptance criteria**

- `pre-commit run --all-files` succeeds on a clean checkout.
- New contributors can set up hooks from the README.

---

## Milestone 2 — Retrieval Quality

### Issue 3: Expand retrieval evaluation dataset

**Type:** `enhancement`  
**Labels:** `evaluation`, `retrieval`, `quality`  
**Priority:** High

**Problem**  
The evaluation flow exists, but the sample question set is intentionally minimal.

**Scope**

- [ ] Add a documented format for project-specific eval datasets.
- [ ] Add examples for fact lookup, character lookup, event chronology, and quote-like questions.
- [ ] Add guidance for collecting relevant chunk IDs.
- [ ] Add a small non-private demo dataset that can run in CI or local smoke tests.

**Acceptance criteria**

- `python eval/run_eval.py --questions ...` works with the expanded examples.
- README explains how to create an evaluation file from a private book collection.
- Retrieval metrics are easy to compare before and after ranking changes.

---

### Issue 4: Add configurable retrieval profiles

**Type:** `enhancement`  
**Labels:** `retrieval`, `configuration`, `api`  
**Priority:** Medium

**Problem**  
Different workloads may need different tradeoffs between latency, recall, and answer quality.

**Scope**

- [ ] Define profiles such as `fast`, `balanced`, and `quality`.
- [ ] Map each profile to semantic, BM25, hybrid, rerank, and context settings.
- [ ] Expose profile selection in CLI and API requests.
- [ ] Include the effective profile in API responses for debugging.

**Acceptance criteria**

- Users can choose a profile without editing environment variables.
- Default behavior remains backward compatible.
- Tests cover profile parsing and request handling.

---

### Issue 5: Improve source attribution in answers

**Type:** `enhancement`  
**Labels:** `llm`, `retrieval`, `ux`  
**Priority:** High

**Problem**  
Answers include sources, but users benefit from clearer source references and confidence signals.

**Scope**

- [ ] Add compact source IDs to prompt context.
- [ ] Ask the LLM to cite source IDs in generated answers.
- [ ] Return cited and retrieved sources separately when possible.
- [ ] Add tests for response shape and fallback behavior.

**Acceptance criteria**

- API responses make it clear which chunks supported the answer.
- If the LLM cannot cite sources, the service still returns retrieved sources.
- Existing clients remain compatible.

---

## Milestone 3 — API & Frontend Demo

### Issue 6: Add OpenAPI examples and request validation details

**Type:** `enhancement`  
**Labels:** `api`, `documentation`  
**Priority:** Medium

**Problem**  
FastAPI generates docs automatically, but richer examples would make the API easier to demo.

**Scope**

- [ ] Add example request bodies for `/ask`, `/search`, and `/build`.
- [ ] Add example responses for success and error cases.
- [ ] Ensure validation errors are clear for empty or malformed questions.

**Acceptance criteria**

- `/docs` shows practical examples for the main endpoints.
- Invalid input returns helpful error messages.
- Tests cover at least one validation failure.

---

### Issue 7: Improve frontend demo source display

**Type:** `enhancement`  
**Labels:** `frontend`, `ux`, `api`  
**Priority:** Medium

**Problem**  
The static frontend is useful for demos, but source inspection can be improved.

**Scope**

- [ ] Show source book names, chunk IDs, and retrieval scores clearly.
- [ ] Add collapsible source text blocks.
- [ ] Display request latency and source count.
- [ ] Handle API errors with user-friendly messages.

**Acceptance criteria**

- Users can inspect sources without opening developer tools.
- Frontend remains dependency-free and easy to serve with `python -m http.server`.
- Existing API base URL configuration still works.

---

## Milestone 4 — Observability & Benchmarking

### Issue 8: Add benchmark comparison reports

**Type:** `enhancement`  
**Labels:** `benchmark`, `observability`, `quality`  
**Priority:** Medium

**Problem**  
The project can generate benchmark reports, but comparing two runs would make performance regressions easier to spot.

**Scope**

- [ ] Add a command or script to compare two latency JSON files.
- [ ] Report p50, p95, p99, mean, and regression percentages.
- [ ] Include retrieval metric deltas when eval artifacts are provided.
- [ ] Document the workflow in README or benchmark docs.

**Acceptance criteria**

- A before/after report can be generated from two benchmark runs.
- Regressions are visible in Markdown output.
- The script exits non-zero only when an optional threshold is provided and exceeded.

---

### Issue 9: Add optional Prometheus labels and request counters

**Type:** `enhancement`  
**Labels:** `observability`, `api`  
**Priority:** Low

**Problem**  
The project exposes dependency-light metrics, but additional counters would help local dashboards.

**Scope**

- [ ] Track request counts by endpoint and status class.
- [ ] Track cache hit/miss counts when LLM cache is enabled.
- [ ] Track index build/load counters.
- [ ] Keep implementation dependency-free unless a metrics dependency is explicitly chosen later.

**Acceptance criteria**

- `/metrics` remains Prometheus-compatible text.
- Metrics do not expose private book content or user questions.
- Tests verify metric output contains expected counters.

---

## Milestone 5 — Packaging & Deployment

### Issue 10: Add release checklist

**Type:** `chore`  
**Labels:** `release`, `documentation`  
**Priority:** Low

**Problem**  
The project has packaging metadata, but there is no repeatable release process.

**Scope**

- [ ] Add a release checklist to the repository.
- [ ] Include version bump, changelog, tests, Docker build, and smoke-test steps.
- [ ] Document how to verify CLI, API, frontend demo, benchmark, and eval flows.

**Acceptance criteria**

- A maintainer can follow the checklist without guessing the release steps.
- The checklist separates required and optional release tasks.

---

### Issue 11: Add production deployment notes

**Type:** `documentation`  
**Labels:** `deployment`, `security`, `documentation`  
**Priority:** Low

**Problem**  
The app is local/offline-first, but users may still want guidance for private LAN or single-user deployment.

**Scope**

- [ ] Document recommended local-only and private-network setups.
- [ ] Explain CORS, API exposure, private data risks, and generated index storage.
- [ ] Add reverse proxy notes without committing to a specific cloud provider.
- [ ] Document backup and cleanup recommendations for `CHUNKS_DIR`.

**Acceptance criteria**

- Documentation clearly warns against exposing private book collections publicly.
- Deployment notes are practical for local and private-network usage.

---

## Backlog

### Issue 12: Add streaming answers

**Type:** `enhancement`  
**Labels:** `api`, `llm`, `frontend`  
**Priority:** Medium

**Scope**

- [ ] Add streaming support to the LLM client when the provider supports it.
- [ ] Expose a streaming API endpoint or optional request flag.
- [ ] Update the frontend demo to render partial answers.
- [ ] Preserve source metadata in the final response.

---

### Issue 13: Add multi-collection support

**Type:** `enhancement`  
**Labels:** `retrieval`, `configuration`, `api`  
**Priority:** Low

**Scope**

- [ ] Allow multiple named book collections.
- [ ] Store indexes and manifests per collection.
- [ ] Add CLI and API collection selection.
- [ ] Prevent cross-collection source leakage.

---

### Issue 14: Add import progress reporting

**Type:** `enhancement`  
**Labels:** `cli`, `api`, `ux`  
**Priority:** Low

**Scope**

- [ ] Report indexing progress by file and stage.
- [ ] Add clearer build summaries.
- [ ] Consider API polling for long-running index builds.

