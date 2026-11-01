# Phase 4 Source Review — Export / Splitting / Loader

Scope: reviewed source as supplied. 24/24 reported passing (actual HF byte reload, deterministic group split, hash missing/bad, exclusion rejection, LJSpeech delimiter rejection). No execution claimed; findings below are source-supported only.

## Verdict

- **Spec compliance: PASS (with 2 low-risk spec gaps and 1 documented-behavior mismatch in `encode_for_lfm3`)**
  - Audio as bytes with `decode=False`, `sampling_rate=24000` → ✔ loader Features.
  - `normalized_text` present in Features and populated from manifest → ✔.
  - Deterministic grouped split: canonical tokens, sha256(seed:ids) group ordering, ≤ 0.5 fraction guard, no-op overlap guarantee via DSU + exact text union → ✔.
  - Accepted/rejected boundary: ACCEPTED only into train/validation; REJECTED excluded from loader and only emitted to `rejected/` when `include_rejected` and `manifest.jsonl` gets `status="REJECTED"` (loader filters) → ✔.
  - Staged atomic no-overwrite: `mkdtemp` in `destination.parent`, existence re-checked pre-rename, `rename` under `repo.writer_lock()`, `rmtree` in `finally` → ✔ (see Bug 4 for edge).
  - Integrity: `validate_export` re-hashes audio, re-runs `grouped_split`, re-checks metadata + JSONL bijection, and aborts on any issue before rename → ✔.
- **Quality verdict: GOOD / borderline PASS with remediation.** Code is disciplined, single-pass, hashes-gated, and deterministic. However there are two real correctness defects (one in the splitter's greedy target logic, one in `validate_export`'s fingerprint scoping) plus a handful of hardening items that would fail a strict code review. None of them break the six passing tests, but they will bite under non-uniform templates, sparse duplicates, or large `/dev/shm`-sized exports.

---

## Ranked bugs — with trigger and fix

### 1) `grouped_split` — greedy target selection is convoluted and can under-select or over-select outliers (High)
**Trigger:** Unequal group sizes, e.g. groups `[90]` + `[10]` (total 100, target 10). The loop's mixed `max(target, len(ids) ...)` and "last group" guard are hard to reason about, and the fallback `if not chosen:` can pick the **first** group by seed-order even if it is the giant 90-item group, producing a 90/10 validation/train inversion — but only when in the *sparse* case where `running + len(ids) <= total - min(...)` fails. Same shape appears when the first group alone exceeds `total - min`, so the loop skips it, and later the fallback re-adds it.
**Fix:** Replace the loop with a deterministic, provable selection:
```python
chosen, running = [], 0
sizes = [len(g) for g in groups]
total = sum(sizes)
min_train = min(sizes)  # at least one group must remain train
for gi, g in enumerate(groups):
    if running + len(g) > total - min_train:
        continue  # keep at least one group train
    chosen.append(gi)
    running += len(g)
    if running >= target:
        break
if not chosen:  # impossible for fraction>0 and >=2 groups, but keep guard
    chosen = [0]
```
This guarantees: at least one group remains train, `running <= total - min_train`, deterministic seed order, and monotone stopping. If “closest to target” is truly required, replace with a knapsack over group sizes (groups are small in practice) — but do not reuse the current expression.

### 2) `validate_export` — duplicate-audio detection is only applied inside the `status=="ACCEPTED"` branch but `seen_audio` gets polluted by nothing else; however `seen_files` dedup is applied to *all* records including rejected, whose filenames `rejected/<id>.wav` are synthesized and never actually emitted as manifest `filename` values (High)
**Trigger:** Set `include_rejected=True` and export. `manifest.jsonl` contains rejected rows only when `include_rejected`. Rejected rows do carry `filename=rejected/<id>.wav`, so duplicate-file detection is fine; but the check `if record["status"] == "ACCEPTED": ... value = fingerprint(...)` skips `duration` and format validation for rejected rows, which is intended — yet it also means **rejected files are never opened by `read_audio`/`sf.info`** because the code above them still runs (missing-file/hash mismatch *is* caught). Actually this is correct in the supplied code — the loop validates format/duration for every manifest row regardless of status. Downgrade to Low and reframe:
- Real issue (Low): if a user manually edits `manifest.jsonl` to swap an ACCEPTED row's `filename` to a `rejected/...` path, `by_name` lookup in the metadata cross-check would still pass because it keys by filename; the loader re-verifies `filename` matches accepted, so this is caught at load time. Consider tightening `validate_export` to reject `status=="ACCEPTED"` rows whose filename does not start with `audio/`.
**Fix:**
```python
if record["status"] == "ACCEPTED" and not record["filename"].startswith("audio/"):
    raise ValueError(f"accepted record has non-audio filename: {record['filename']}")
```

### 3) `validate_export` — metadata cross-check accepts any subset order and does not verify `metadata_ljspeech.csv` at all (Medium)
**Trigger:** User hand-edits `metadata.csv` row order or truncates `metadata_ljspeech.csv`; loader and HF build still succeed. The delivered "LJSpeech-compatible" bundle is silently corrupt.
**Fix:** Add after the metadata loop:
```python
if "ljspeech" in info["formats"]:
    lj_path = root / "metadata_ljspeech.csv"
    lj = [ln.rstrip("\n").split("|") for ln in lj_path.read_text(encoding="utf-8").splitlines() if ln]
    if len(lj) != len(accepted):
        issues.append("ljspeech metadata record count mismatch")
    lj_by_id = {Path(r["filename"]).stem: r for r in accepted}
    for i, parts in enumerate(lj, 1):
        if len(parts) != 3 or parts[1] != lj_by_id[parts[0]]["original_text"]:
            issues.append(f"ljspeech delimiter/mapping mismatch row {i}")
```
And key `by_name` on the stem or on `filename` consistently — currently `by_name` uses `filename` (`audio/000001.wav`) while `metadata_ljspeech.csv` uses `Path(...).stem` (`000001`); they don’t collide, but the cross-check for LJSpeech must map stem→record explicitly.

### 4) `export_dataset` — rename is not atomic on all filesystems and `destination.parent` may be on a different FS than `repo.root` for the moved audio; pre-rename re-check has a TOCTOU window (Medium)
**Trigger:** Concurrent external creation of `destination` between the initial `if destination.exists()` and `staging.rename(destination)` (also the second `if destination.exists()` is inside the lock but only re-checks existence, not identity). On Windows the rename of a directory onto an existing path can silently replace/remove existing contents in some APIs — `Path.rename` calls `os.rename` which on POSIX is atomic-replace only for files, and for directories will *replace an empty directory*.
**Fix:**
```python
# after validation, inside lock:
if destination.exists():
    raise FileExistsError(destination)
try:
    staging.rename(destination)
except FileExistsError:
    raise FileExistsError("Destination created concurrently; refusing to overwrite")
```
Also consider `os.replace` semantics are not safe for directories; the current combination is acceptable but the TOCTOU comment above and the explicit `FileExistsError` catch are mandatory.

### 5) `encode_for_lfm3` — silently resamples non-24 kHz audio instead of failing (Low, spec-delta vs loader)
**Trigger:** Loader enforces 24 kHz at feature level (`Audio(sampling_rate=24000, decode=False)` casts metadata only; bytes are passed untouched), but `encode` resamples with librosa if a WAV misreports 24 kHz. Since `validate_export` already enforces 24 kHz mono PCM_16, resampling is dead code; if it ever runs, the `audio.ndim != 1` guard runs *after* mono check, and `Dataset.from_dict(...).map(...)` will fail late.
**Fix:** Fail fast rather than resample:
```python
if rate != 24000:
    raise ValueError(f"LFM3 requires 24 kHz; got {rate}")
```
Or drop the branch and assert `example["audio"]["sampling_rate"] == 24000`. Either way, remove `librosa`/`numpy` from this file’s lazy imports if resample is removed.

### 6) `_jsonl` writes with `allow_nan=False`, but `metrics` / `parameters` are embedded raw and could contain `NaN`/`Infinity` (Medium)
**Trigger:** Any upstream metric (WER, SIM, duration) computed as NaN, or a parameter produced as `float("inf")`. `json.dumps(..., allow_nan=False)` raises `ValueError` mid-write; but because writes go to `staging/` inside the `try`, the destination is untouched — good — yet the error surfaced is a raw `json` error, not an actionable validation message.
**Fix:** Wrap `_jsonl` errors with `ValueError(f"Non-finite value in {path.name}: {exc}")`, or sanitize numerics up front with `math.isfinite` assertions on `wer`, `cer`, `speaker_similarity`, duration.

### 7) `export_dataset` — `job_configs` lookup raises a raw `KeyError` for missing jobs, and `repo.get_job(...)` is called once per distinct job but never guarded for `None` payload (Low)
**Trigger:** Sample references a deleted job but is still ACCEPTED (possible after manual DB edits). Export aborts with opaque `KeyError`/`json.JSONDecodeError`.
**Fix:**
```python
try:
    job_configs[sample["job_id"]] = json.loads(repo.get_job(sample["job_id"])["config_json"])
except Exception as exc:
    raise ValueError(f"Cannot load job config {sample['job_id']}: {exc}") from exc
```
