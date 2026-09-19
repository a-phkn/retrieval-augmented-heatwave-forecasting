# Context Folder

This folder is the project's persistent memory. It exists so a future Claude
session (or a human collaborator) can understand the project's current state
without re-uploading the entire repository or re-deriving decisions that have
already been made.

## What each file contains

- `project_overview.md` — what the project is, architecture, datasets, current state, at a glance.
- `build_plan.md` — the roadmap broken into milestones with status tracking (NOT STARTED / IN PROGRESS / COMPLETED / BLOCKED).
- `progress.md` — short, frequently-updated running tracker of what's done/in-progress/next.
- `architecture.md` — what the code actually does (directory structure, data flow, interfaces), not an idealized design.
- `data.md` — dataset-by-dataset documentation of every file under `data/`, `datasets/`, `events/`.
- `ml_notes.md` — accumulated ML knowledge: models tried, hyperparameters, seeds, actual results.
- `decisions.md` — architectural/ML decision log, including flagged issues and how they were resolved.
- `environment.md` — how to reproduce the dev environment locally and on Colab.

## Future Session Protocol

At the start of a new session, read in this order:

1. `context/README.md` (this file)
2. `context/project_overview.md`
3. `context/progress.md`
4. `context/build_plan.md`
5. Any specialized context file relevant to the task at hand (`data.md`, `ml_notes.md`, `architecture.md`, `decisions.md`, `environment.md`)
6. Then inspect the actual source code relevant to the task — **do not trust this folder blindly**. If code and context disagree, the code is the source of truth; fix the context file.

## Maintenance rule

Update the relevant file(s) whenever a meaningful implementation milestone completes or a decision is made. Never mark something COMPLETED in `build_plan.md` without having actually run and verified it.
