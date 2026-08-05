# SecureDocQA — Standing Rules for Claude Code

You are implementing an approved design. Execute it. Do NOT invent
architecture. If something in a task contradicts the design, STOP and ask —
do not improvise.

## Project
- Python 3.12, Windows dev, deploys to Render (Linux).
- FastAPI modular monolith. Modules: app/api, app/security, app/pipeline,
  app/retrieval, app/llm, app/session. Config in app/config.py (`settings`).
- Frontend: vanilla HTML/CSS/JS, no build step.

## Hard rules (every task)
- One component at a time. Build ONLY what the task asks. No extra endpoints,
  no extra dependencies, no "helpful" additions. Scope creep is a defect.
- Every function that receives external data validates its input.
- Every function that returns data to a user or store validates its output.
- Wrap EVERY external call (LLM, Qdrant, file I/O, network) in try/except and
  raise a typed custom exception. No raw library exception may reach a caller.
- A timeout on every network call. No exceptions.
- Every retry is bounded. No unbounded loops.
- No bare `except:`. `except Exception` is allowed ONLY when parsing
  untrusted input, and only if it re-raises as a typed error with `from exc`.
- The user NEVER sees a stack trace, file path, model name, or raw exception.
  Log full detail server-side; return a clean typed error.
- Never hardcode secrets or model IDs. Read everything from `app.config.settings`.
- Cyclomatic complexity < 10 per function (ruff C901 enforces this).

## Testing
- Every component ships with pytest tests in the same task.
- Tests must not depend on external files or network — generate fixtures
  in-memory.
- After writing: run `pytest` (ALL tests must pass) and `ruff check .`
  (no errors). Report both results.

## Self-review before finishing
Before handing back, check your work against this file and list any
deviations. Confirm: scope not exceeded, all external calls wrapped, timeouts
present, no secrets/model-IDs hardcoded, complexity under 10, tests pass.

## Never do
- Never commit. The human reviews the diff and commits.
- Never run git commands that change history.
- Never add a dependency without stating why in your summary.

