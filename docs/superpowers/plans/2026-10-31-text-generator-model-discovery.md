# Text Generator clarity and model discovery

Clarify built-in templates, OpenAI-compatible LLM text, and imported text; explain
voice language, preview count versus accepted-audio target, finite template variety,
relative category weights, topics, word filtering, and where to review sentences.
Show only relevant source options, preserving values when switching modes.

Replace the LLM model line edit with an editable discovered-model combo. Reuse the
existing endpoint validation and aux TaskRunner for GET /models. Debounce endpoint
edits and auto-refresh when using the LLM source on Text Generator, with a manual
Refresh models button. Preserve saved/custom IDs, reject stale results after endpoint
or permission changes, respect remote opt-in, and show failures beside the dropdown.
No network at startup in built-in mode; no extra workers or changes to generation.

- [x] Add failing real-loopback HTTP tests for discovery, bad/empty replies,
  redirects and endpoint restrictions.
- [x] Add shared endpoint validator and bounded model-ID parsing.
- [x] Add Qt tests for automatic fill, manual selection/config persistence,
  existing preview generation, source controls and stale/error results.
- [x] Implement source explanations, relevant option groups and editable model
  dropdown; update the two settings bindings.
- [x] Update guide/docs, run full suite and lint, visually check the page and review.

Completion: 126 tests passed; targeted Ruff clean. Reviewed normal/compact layouts;
the LLM form and built-in form fit 1000×700 without horizontal scrolling. Independent
read-only review found no actionable issues. Discovery/preview were tested against
real loopback fixture servers, with no requests to the user's LLM server. Existing
generation, settings, remote permission and manual model-ID support are retained.
