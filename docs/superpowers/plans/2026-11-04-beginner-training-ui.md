# Beginner training and voice preview implementation plan

**Goal:** Implement the approved guided training screen and accessible Chatterbox previews.

**Spec:** `docs/superpowers/specs/2026-11-04-beginner-training-ui-design.md`.

**Approach:** Work inline on the existing local installer branch. Reuse existing
controllers, worker runners, saved widget bindings and media playback. Preserve
the uncommitted installer work. Do not commit, push, install dependencies, download
models, or run real training/generation without a further request.

## Tasks

1. Write failing behavioral tests for training source/readiness, stale asynchronous
   summaries, progression, force-stop cancellation, shared preview settings,
   custom text, immutable references and background execution.
2. Add a small reusable expandable UI section and a read-only training-source
   summary helper. Organize training into data, setup and run sections; disable
   unavailable actions with a reason, show real progress, retain logs and advanced
   fields, and preserve all existing controller semantics.
3. Reorganize Voice Manager around the existing synthesis settings and preview
   mechanism. Share its snapshot construction with existing configuration saving;
   permit project-free previews and provide playback plus clear preview-only status.
4. Wire contextual training help and preview shortcuts. Keep page names/settings
   compatible and put all blocking metadata/audio work on owned workers.
5. Run targeted and full tests, lint and visual QA at normal/narrow sizes. Request
   one independent review, resolve material findings, update user instructions,
   and rebuild the installer and checksum. Deliver uncommitted local changes.

## Review focus

- A source changed during validation must discard stale results.
- Training/preview/generation and metadata readers must not contend for writers.
- Preview settings are captured before dispatch; controls edited during a preview
  must not change its voice, output project, or parameters.
- Saved reference hashes and job settings must not change implicitly.
- Invalid data, unavailable environments, busy workers and failed processes must
  explain the next action without claiming completion or losing available outputs.

## Progress

- Existing installer baseline: 153 tests passed.
- Current UI was captured and inspected before changes; no real project was opened.
- Tasks 1–5 complete locally: guided training, shared Chatterbox preview controls,
  contextual help, instructions, behavioral tests, and rebuilt installer.
- Final verification: 180 tests passed in 43.87 seconds; Ruff and whitespace checks passed.
- Visual QA: both changed pages inspected at 1250 x 850 and 1000 x 780, without
  horizontal overflow; the preview action fits the normal viewport and narrow
  sliders remain usable. Expandable settings/logs work with keyboard Space.
- Independent review found two P2 issues. Both were reproduced with failing
  tests and fixed: reconfirm permission on reference edits, and rediscover
  compatible models for an English-to-Swedish quick-preview handoff.
- Added a regression preventing periodic diagnostics from repeatedly hashing an
  unchanged dataset export. Preview settings remain authoritative for new Batch
  Generator jobs; Getting Started retains its existing automatic dataset defaults.
- The updated installer contains 69 payload files, including the theme SVGs, and
  a verified SHA-256. No actual GPU training/synthesis, model downloads, or full
  dependency installation were performed for this UI task.
- The user subsequently authorized committing and pushing all changes and
  publishing the updated Windows installer in release v0.1.0.
