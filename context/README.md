# Context Folder

This folder is the project's persistent memory. It exists so a future Claude
session (or a human collaborator) can understand the project's current state
without re-uploading the entire repository or re-deriving decisions that have
already been made.

## What each file contains

**Current (kept up to date):**
- `progress.md`: the running tracker of what's done, in progress and next.
- `decisions.md`: the dated decision log. Every experiment's rule is written here before it is run.
- `environment.md`: how to reproduce the environment (project venv, Python 3.13).

**Historical (v1, September to early October 2026; kept because the frozen code refers to them):**
`project_overview.md`, `build_plan.md`, `architecture.md`, `data.md`, `ml_notes.md`, `RUN_COMMANDS.md`.
Each one starts with a note saying so.

**Start here instead:** `../docs/PROJECT_GUIDE.md` (the plain-language guide to the whole project, kept current),
`../docs/MODELS_EXPLAINED.md` (every model and metric), `../docs/CONFIRMATORY_PROTOCOL.md` (the frozen test
protocol) and `../docs/GLOSSARY.md` (every short form).

## Future Session Protocol

At the start of a new session, read in this order:

1. `context/README.md` (this file)
2. `docs/PROJECT_GUIDE.md`
3. `context/progress.md`
4. `context/decisions.md`, for the part relevant to the task
5. Then inspect the actual source code relevant to the task. **Don't trust these notes blindly**: if code and notes
   disagree, the code is the source of truth, and the notes get fixed.

## Maintenance rule

Update `progress.md`, `decisions.md` and `docs/PROJECT_GUIDE.md` whenever a meaningful milestone completes or a
decision is made. The historical files are not updated.
