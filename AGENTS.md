# Agent instructions

Standing rules from the repository owner for every AI agent working on this project (Claude Code, Codex, Cursor, pi). Keep this file identical in both repositories.

## Communication

- Get right to the point.
- Don't be a yes man. If the owner is wrong, say so.
- Judge the owner's suggestions on their merits; don't lean toward one just because the owner made it. Account for every possibility and aspect.
- Change position only when given a reason, not because the owner pushes back. Don't be stubborn without a reason either: be reasonable and consistent.
- Never use em dashes.

## GitHub

- Do all GitHub work (commits, pushes, pull requests, comments) through the GitHub MCP server, as the owner's account.
- Never add `Co-Authored-By`, `Claude-Session` or "Generated with Claude Code" lines to commits or pull request descriptions.

## Branches and pull requests

- Commit to a branch as you go.
- Open a pull request only once the work is fully finished and tested. Never open draft pull requests.
- If a pull request gets review findings or is rejected, fix them on the same branch and push to the existing pull request. Never open a new pull request for fixes.

## Project

- `itscool2b/ig-work-2026-private-full` (private) holds every experiment since the project began, plus the research notes. `itscool2b/ig-response-geometry` (public) is the published subset: paper, saved data and analysis code.
- `notes/` in the private repo documents the whole project, Month 1 onward. Start with `notes/00 Project Narrative.md` and the newest month folder.
- Document everything you do on this project in the private repo's `notes/`, the way earlier months are documented: numbered notes in the current month's folder following `notes/Style Guide.md`, plus dated entries in that month's execution log.
- Agents do the code, experiments, data recording and analysis. The owner writes every word of the manuscript and the reviewer response, captions included; agents never draft or edit that text.
- GPU work runs on the owner's RunPod account. Cloud sessions have no GPU and cannot SSH, so they reach RunPod through its HTTPS API with the key in `RUNPOD_API_KEY`.
