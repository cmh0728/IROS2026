# Repository Guidelines

## Communication Language

Always communicate with the user in Korean. Write progress updates, explanations, questions, summaries, and final responses in Korean. Keep code, identifiers, commands, file paths, API names, configuration keys, and conventional technical terms in English where appropriate. Comments and documentation must follow the existing project style unless the user explicitly requests Korean. The fact that this guide is written in English does not change the Korean-response requirement.

## Workspace Scope & Repository Boundaries

This root guide applies to both independent Git repositories in the workspace:

- `earth-rovers-sdk/`: FastAPI/Hypercorn service for rover control, telemetry, cameras, and speech. Core modules are at the project root, browser assets are in `static/`, and runnable samples are in `examples/`.
- `Asl-prototype/earth-rover-hybrid-autonomy/`: installable Python autonomy package. Production code lives under `src/earth_rover/`, grouped by `control`, `navigation`, `perception`, `planning`, and `safety`. Tests mirror those responsibilities in `tests/`; runtime profiles live in `configs/`; entry points are in `scripts/`.

Each repository has independent history, dependencies, commands, and working state. Before editing, identify the owning repository and run Git commands from its root. Never infer one repository's status from the other, initialize Git at the shared workspace root, or modify both repositories unless genuine cross-repository integration requires it. Before a cross-repository edit, explain in Korean why both must change.

Search from the target file toward the repository root for a closer `AGENTS.md`; its more specific rules apply within that subtree in addition to this guide. If rules conflict, follow the closest applicable file unless a higher-priority instruction says otherwise.

Treat `datasets_2k/`, `datasets_7k/`, generated `logs/`, `screenshots/`, caches, and virtual environments as local artifacts, not source.

## Required Development Environment

Use the Dell workstation running Ubuntu 22.04 as the sole development and test environment for both repositories. Perform code changes, dependency installation, automated tests, dataset processing, CUDA inference and training, SDK dashboard validation, replay, and rover integration on that host. Keep commands and documentation Linux-specific and do not add platform-specific workflows for other operating systems.

Hardware-dependent validation must record whether it used the Dell GPU or a live rover. Development on the Dell workstation does not itself authorize live motion; the rover safety rules below still apply.

## Project Direction & Sources of Truth

The primary target is Track 01 Urban GPS navigation. The intended architecture is a hybrid autonomy stack: learned models primarily support perception, traversability, risk estimation, or related decision inputs, while classical navigation and control plus rule-based safety, recovery, logging, replay, and SDK integration remain explicit. Do not assign learned models another role or convert the stack into end-to-end learned control without explicit user approval.

Keep SDK communication, sensor processing, goal management, perception, planning, control, safety, recovery, and logging reasonably separated. Improve the existing Urban MVP incrementally instead of creating a parallel architecture or recreating completed behavior under new names.

Current implementation, tests, configuration, and Git state take precedence over old plans, conversation memory, status summaries, and documentation. Documentation is important context but may be stale. Before proposing a component, search for an implementation, configuration, test, script, or partial version. Never recreate completed functionality under another name. When sources disagree, report the inconsistency and determine which source controls runtime behavior; do not silently update code or docs unless the request includes that change.

**Historical checkpoint, always verify:** SDK setup and client/endpoint integration, GPS utilities, waypoint management, controller, logger, the basic Urban MVP, console status output, the two-second latency profile, delayed replay, and rear-camera recovery logging have existed. This is orientation, not proof of current status. Do not propose these as new primary tasks without repository evidence of regression, missing integration, or an explicit request.

## Session Continuity

At the start of new or resumed work, reconstruct relevant state from the repository rather than relying on remembered conversation context. Inspect the applicable `AGENTS.md`, correct repository root, `git status`, existing diffs, recent relevant commits, code and tests, current configuration, and any applicable status or handoff document.

Preserve unfinished changes. When reasonably possible, determine whether they belong to the user, a prior Codex session, or the current task. Continue from them only when relevant; never overwrite or revert them. Ask only if ownership or intent is materially ambiguous and proceeding could destroy or conflict with work. `/init` is not a progress-saving command; do not rerun it merely because a day or session changed.

## Build, Test, and Development Commands

Run commands from the relevant project directory.

```bash
# SDK service
pip3 install -r requirements.txt
hypercorn main:app --reload                 # serves http://localhost:8000
docker compose up --build                   # containerized alternative

# Hybrid autonomy
pip install -r requirements.txt
pytest                                      # runs tests/ via pyproject.toml
python scripts/run_sdk_smoke_test.py --config configs/default.yaml --no-motion
python scripts/run_urban_mvp.py --config configs/default.yaml
```

Use `--no-motion` for initial SDK checks. Running the MVP can issue real rover commands and requires a configured SDK service.

## Coding Style & Naming Conventions

Use Python 3.10+ for autonomy code and four-space indentation throughout. Follow PEP 8: `snake_case` for modules, functions, and variables; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. Add type hints to public interfaces and keep imports grouped as standard library, third party, then local modules. Inspect repository configuration for formatter and linter rules; when none apply, preserve the surrounding style and keep changes narrowly scoped.

## Task Classification & Autonomy

Classify the task before acting: explanation/review, diagnosis, implementation, refactoring, live-rover operation, dataset operation, or monitoring/waiting.

- For explanation, review, or diagnosis, inspect and report without editing unless the user requests a change. Diagnosis identifies cause and a proposed fix; it does not imply implementation.
- For implementation, complete only the requested behavior and verify proportionally. For refactoring, establish relevant behavior or tests before and after.
- For monitoring or waiting, observe and report; do not introduce unrelated changes.
- Trivial edits need no elaborate plan or excessive commentary. Non-trivial work needs a short Korean plan with observable checks. Safety-critical, control, architecture, live-rover, and dataset tasks require stronger inspection and verification.

Ask only when missing information materially changes architecture, safety, live-rover behavior, public interfaces, data handling, destructive actions, or the expected result. Prefer read-only repository inspection before asking for information available locally. For low-risk ambiguity, choose the smallest reversible option, state the assumption briefly in Korean, and continue. When several approaches are valid, recommend one and explain the decisive tradeoff instead of presenting an unranked list.

Push back when a request weakens safety, duplicates existing work, adds unjustified complexity, or conflicts with the hybrid architecture. Never invent SDK behavior, endpoint schemas, dataset structure, test results, hardware behavior, or competition requirements.

## Engineering Workflow

### Think Before Coding

Inspect relevant call paths, data flow, code, tests, configuration, documentation, and repository state before editing. Search before creating any file, class, module, abstraction, configuration key, or script. Never infer that a component is absent from an outdated plan alone.

Plans must state outcomes and checks, not vague activities. A useful pattern is:

1. Inspect existing behavior and reproduce the issue. Verification: failing test, replay case, log evidence, or documented observation.
2. Make the smallest targeted change. Verification: the focused check passes.
3. Check for regression. Verification: the relevant suite or replay passes.

Update the plan when repository evidence invalidates an assumption; never follow an outdated plan merely because it was written earlier. Stop for a blocker only after exhausting safe, in-scope inspection and alternatives.

### Simplicity and Surgical Changes

Implement only the requested behavior using the smallest explicit, readable, testable, and safe change. Extend existing modules before adding replacements. Do not add speculative features, unused configurability, premature frameworks, unnecessary abstractions, or single-use wrappers. Never compress safety-critical logic merely to reduce line count.

Change only files and lines required by the task. Do not broaden work into adjacent improvements, reformat, rename, refactor, or clean up unrelated code. Preserve comments, public interfaces, and behavior unless directly affected, and match existing style and structure. Avoid unrequested parallel implementations, compatibility layers, broad dependency changes, and speculative handling for impossible or unsupported cases. Use the standard library or an existing dependency where sufficient; explain and obtain confirmation before adding a substantial dependency or changing an external interface.

Mention unrelated dead code or debt without modifying it. Remove only imports, variables, functions, tests, or files made unused by the current change. Preserve user edits and unrelated uncommitted work. Before finishing, inspect the diff and ensure every changed line is traceable to the request; remove accidental formatting, debug code, temporary files, secrets, and generated artifacts.

### Goal-Driven Execution

Translate vague requests into observable success criteria. For bug fixes, first reproduce the issue with the smallest practical automated test, replay fixture, log sample, or deterministic simulation. If reproduction is impractical, document the evidence and a concrete manual procedure. Iterate until the criteria pass or a genuine blocker is identified; never claim completion for partial implementation or unverified behavior.

## Rover & Operational Safety

Never command a live rover unless the user explicitly authorizes live movement for the current task. Discussion, code preparation, configuration review, dry runs, and no-motion smoke tests are not authorization. Authorization from a previous task or session does not carry forward. Never silently switch from offline or dry-run execution to live control or run live-motion tests unattended.

Before any authorized live-motion test, state in Korean what will move, expected command ranges, stop conditions, and how the user can interrupt it. Begin with the lowest practical command and shortest practical duration. Stop immediately on stale telemetry, communication loss, unexpected motion, invalid sensor input, or user interruption.

Default to unit tests, mocks, recorded logs, offline or delayed replay, simulation, SDK dry runs, and no-motion modes. Clamp outgoing `linear` and `angular` values to the SDK-supported range. Missing, stale, malformed, delayed, or inconsistent critical sensor data must produce safe, explainable behavior. Preserve or strengthen emergency-stop and fail-safe behavior; never bypass a safety check to pass a test. Never expose API keys, tokens, credentials, rover identifiers, or `.env` contents in logs, diffs, commands, or responses.

## Dataset Safety

Before a large download or transformation, inspect dataset size, format, license constraints, storage availability, extraction requirements, and archive structure. Prefer metadata, manifests, dry runs, or small verified samples before full processing; never assume an archive supports partial sampling.

Do not destructively delete, overwrite, move, recompress, or re-extract large datasets without explicit authorization. Keep generated indexes, caches, checkpoints, and logs out of Git unless intentionally maintained as source artifacts. Treat FrodoBots-2K and Berkeley-FrodoBots-7K as distinct until their actual schemas and pipeline compatibility have been verified.

## Testing & Verification

Autonomy tests use `pytest`; place tests under `tests/`, name files `test_<behavior>.py` and functions `test_<expected_result>`, and reuse fixtures from `tests/conftest.py`. Avoid network, hardware, and motion dependencies in unit tests. Inspect the current SDK test setup before choosing verification; do not rely on a permanent assumption about whether an automated suite exists.

Distinguish static inspection, focused automated tests, full automated suite, offline replay, delayed replay, simulation, SDK dry run, manual endpoint verification, and live-rover verification. Never claim behavior works from code inspection alone, fabricate or infer passing results, or claim live validation without authorized hardware testing performed during the task.

Prefer the narrowest relevant check first, then broader regression checks proportional to risk. For navigation, control, recovery, SDK, or sensor processing, consider recorded replay, delayed replay with `configs/urban_latency_2s.yaml`, stale/missing/malformed/dropped inputs, command ranges, stop/recovery behavior, log/timestamp consistency, and dry runs. Do not run unrelated expensive tests.

Never weaken, delete, skip, or rewrite a valid test merely to pass. Distinguish pre-existing failures from regressions caused by the change. Report each executed command and its exact summarized result. If verification is blocked by environment, dependencies, hardware, network, data, or credentials, state the precise limitation and what remains unverified. Report manual checks in the Korean completion summary; include them in a PR description only when the task actually involves a PR.

## Commit & Pull Request Guidelines

Each repository has its own Git history; commit from the repository you changed. Existing commits use short, imperative summaries such as `Add lamp in readme` and `Harden ignored secret files`. Keep one logical change per commit. When a task includes a PR, explain behavior and safety impact, list verification, link relevant issues, and include screenshots for web UI changes. Never commit `.env`, API tokens, certificates, rover identifiers, or generated camera data; use the repository's environment example as the configuration template.

## Completion & Handoff Reporting

For completed coding work, report briefly in Korean: outcome, changed files, important behavior or safety impact, verification commands and exact summarized results, what was not verified, and remaining risks or blockers. Do not repeat the plan or claim completion when only part of the requested behavior exists.

For incomplete work, report what is complete, what remains, why work stopped, and the safest next action. Do not create or update a handoff/status file unless requested or required by an existing applicable project rule.
