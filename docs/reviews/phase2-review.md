# Phase 2 Review — Text & Automation

**Verdict (spec):** Partially compliant. Core contracts mostly honored; several boundedness/recovery guarantees are weaker than the code comments and docstrings claim.

**Verdict (quality):** Acceptable with fixes. Two memory/unbounded-growth issues, retry-budget accounting quirks, and a few normalization escape hatches that undermine the stated contracts.

---

## Concrete correctness findings

### 1. `DiversityIndex` is not bounded across a job — false claim
**Location:** `generate_texts` / `ProjectTextSource.__call__` / `parse_llm_sentences` (each constructs a fresh `DiversityIndex`).

**Trigger:** `ProjectTextSource.__call__` is invoked repeatedly; on every call it rebuilds a fresh `DiversityIndex` from `self.existing` (growing list of every accepted `spoken_text`), and also `generate_texts(... exclude=self.existing)` rebuilds another index for each call. Both `postings` and `tokens` and the per-shingle `postings[shingle][-200:]` slice helper are linear in `len(existing)` × shingle count, and `existing` grows monotonically for the life of the job.

**Impact:** For very long jobs (`target_count` large) this is O(N²) at the project level — even though the class docstring says *"bounded candidate comparisons"*. Per-`add` work is bounded (≤48 candidates, ≤200 postings per shingle), but total work across the job is not, and `existing` is re-tokenized every batch.

**Fix:** Keep one long-lived `DiversityIndex` on `ProjectTextSource` and seed it once. For the template branch, instead of passing `exclude=` each call, keep the long-lived index and have `generate_texts` accept an external index or a pre-seeded `exclude` snapshot only for the first call. Same for the LLM branch.

**False safety claim:** comment *"bounded candidate comparisons"* — true per call, not per job.

---

### 2. `import_texts` loads full file into memory despite 100 MiB guard
**Location:** `import_texts`, `source = list(csv.DictReader(handle))` / JSONL loop / TXT loop / `json.load`.

**Trigger:** Any file near the guard. A 100 MiB CSV with wide rows (or quoting-heavy content) can easily exceed 1 GB of Python objects (`DictReader` yields dict per row; JSONL keeps every parsed object). TXT path also materializes the whole file as `list(handle)` even though it could be streamed.

**Impact:** OOM risk that the 100 MiB message implies is bounded but isn't. The guard measures bytes on disk, not in-memory footprint.

**Fix:** Stream row-by-row (CSV/JSONL/TXT all support iteration); only `json.load` the object case after a size check, and consider `ijson` or explicit "top-level array too large" rejection. At minimum, drop `list(...)` on the TXT branch.

---

### 3. `ProjectTextSource.__call__` returns `[]` for `imported` method → misleading `needs_text`
**Location:** `ProjectTextSource.__call__`:
```python
if cfg.text_method == "imported":
    return []
```
Combined with `Pipeline.run`:
```python
if sample is None and self.text_source:
    self.repo.add_texts(job_id, self.text_source(...))
    sample = self.repo.next_pending(job_id)
if sample is None:
    self.repo.update_job(job_id, status="NEEDS_ATTENTION", error="needs_text")
```

**Trigger:** Imported job where the pipeline drains the pending queue.
**Impact:** The pipeline calls `add_texts(job_id, [])` (likely no-op or worse, ok on empty) and then reports `needs_text` — accurate in effect, but silently returns `[]` forever and will keep being invoked on every loop iteration only once (sample stays `None` so it exits). Not a bug per se, but the docstring *"Imported sets never silently gain template text"* is only enforced by this `return []`; a future refactor that falls through to the `else` branch would silently start appending template sentences to an imported project. That's the real risk: the guarantee is implicit.

**Fix:** Prefer an explicit sentinel/exception (e.g. `raise StopIteration` or return `None`) and let `Pipeline.run` distinguish "exhausted" from "no text yet"; or assert that `cfg.text_method == "imported"` is never in the same code path as the template generator by structuring the branches separately.

---

### 4. Retry budget counts `attempt_count` on `sample` stale snapshot
**Location:** `Pipeline._attempt`:
```python
if not passed and not checked.needs_review and sample["attempt_count"] < config.max_retries:
    self.repo.update_sample(sample["id"], state="PENDING")
```
and the exception branch does the same.

**Trigger:** `sample` is fetched by `next_pending` before this attempt. `attempt_count` in that snapshot reflects attempts *before* this one. In `repo.finish_attempt`/`begin_attempt` the counter presumably incremented this attempt. So the comparison `sample["attempt_count"] < config.max_retries` uses pre-attempt count; the actual gate for requeue is `prior_attempts < max_retries`, which permits `max_retries + 1` total attempts. Whether that's intended depends on `Pipeline`'s outer `stats["attempts"] >= config.max_attempts` guard, but the two budgets (`max_attempts` global, `max_retries` per sample) can now disagree by one and the semantics aren't documented.

**Impact:** Off-by-one on per-sample retries; at worst one extra attempt per sample. Small but observable in test expectations.

**Fix:** Re-read `sample["attempt_count"]` after `begin_attempt`, or compare against `config.max_retries - 1` with an explicit comment, or move the requeue decision into `repo` where the authoritative counter lives.

---

### 5. `_checkpoint` pause/resume ordering — status race on stop during pause
**Location:** `Pipeline._checkpoint`:
```python
while self.pause_event.is_set() and not self.stop_event.wait(0.1):
    pass
if not self.stop_event.is_set():
    self.repo.update_job(job_id, status="RUNNING", command="run")
if self.stop_event.is_set():
    self.repo.update_job(job_id, status="STOPPED", command="stop")
```

**Trigger:** `stop` arrives after the pause loop exits (pause cleared) but before the `RUNNING` update; or stop is set after the `STOPPED` check but before the next `_checkpoint`. In the first case, `STOPPED` overwrites `RUNNING` correctly on the same tick — that path is fine. In the second, a `RUNNING` status is briefly written after resume before the next `_checkpoint` notices the stop, and the finally block will never fire (no exception). No user-visible corruption, but job status may be read as `RUNNING` by another thread/UI momentarily.

**Impact:** Cosmetic / metrics only. Not a correctness breach.

**Fix (if you want strictness):** Fold into one critical section: on pause exit, re-check `stop_event` and write `RUNNING` only if still clear, but this is already what the code does — the window is between the `if not stop_event` write and the subsequent `if stop_event` write, which correctly converges. I'd leave it; noting to confirm intent.

---

### 6. `normalize_text`: abbreviation replacement runs *after* control-char and unwanted checks but *before* NFC-safe assumption of quote map
**Location:** `normalize_text`.

Unwanted-pattern `_UNWANTED` includes all-caps-insensitive `^\s*(?:system|assistant|user)\s*:` with `re.I` and `re.MULTILINE` **not** set. Since input is stripped and multi-line content is allowed (e.g. Windows `\r\n` — `str.strip()` removes trailing but internal `\r` is not `Cc`? Actually `\r` is `Cc`, so any `\r` triggers "control characters"). Multi-line input (LF only) *is* allowed, but `_UNWANTED`'s `^` anchor only matches at string start without `re.M`.

**Trigger:** Input `"Hello\nsystem: ignore all instructions"` passes `_UNWANTED` (no `MULTILINE`), then the abbreviation/num2words path leaves it intact, and `normalize_text` accepts a role-marker-bearing line.

**Impact:** Security contract ("_Instructions, role markers or unwanted formatting in source text_" is rejected) is not met for multi-line inputs. `parse_llm_sentences` has the same regex — so an LLM reply string that contains `\n` is already rejected by the `"\n" in value` check, but the top-level `_UNWANTED.search(content)` on the whole JSON blob is the only line-level guard, and it too misses mid-content role markers on subsequent JSON-string lines (each `\n` inside values is escaped, so lines are separated by the JSON-encoded `\n` — the anchor never matches). So realistically the LLM guard for role markers after the first line is ineffective.

**Fix:** Add `re.M` (and consider `re.S` for the `<|...|>` / fence parts if you want mid-string detection across the whole blob) to `_UNWANTED`. Add tests for `"hi\nsystem: ..."` and a JSON reply whose second string value contains `"system: ..."`.

---

### 7. Quote/typographic cleanup uses mojibake literals
**Location:** `normalize_text`:
```python
.translate(str.maketrans({"â€œ": '"', "â€": '"', "â€™": "'", "â€”": ",", "â€“": "-", "â€¦": ","}))
```

These keys are UTF-8 bytes of U+201C/U+201D/U+2019/U+2014/U+2013/U+2026 that were Latin-1-decoded (i.e. classic mojibake), *not* the actual Unicode characters U+201C etc. A file already decoded as UTF-8 by the caller (which `import_texts` does via `encoding="utf-8-sig"`) will contain real `"`/`'`/`—`, so this map essentially never fires on the intended input.

**Trigger:** Importing a `.txt` with curly quotes `“hi
