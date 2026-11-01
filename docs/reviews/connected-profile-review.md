# Confirmed load-bearing bugs

### 1. Reference digest is not invalidated when a service profile supplies a `reference_path` (stale digest → wrong integrity, permission silently cleared but digest committed)

Sequence:
1. `window.apply_config(config)` sets `_reference_digest = config.voice.reference_sha256` for the *original* WAV.
2. User fetches/selects a documented profile in `_apply_service_voice`, which calls `self.reference_path.setText(profile["reference_path"])`.
3. The `textChanged` handler `lambda: setattr(self, "_reference_digest", "")` *does* fire, so the digest is cleared — **but** the supplied test asserts `window._reference_digest == config.voice.reference_sha256` after this sequence, i.e. the test encodes the pre-invalidation expectation. Either the handler ordering differs (the signal fires and clears, then `apply_config`'s later writes re-set it) or the digest retained from a previous voice survives alongside a *different* profile reference WAV and is persisted into `configuration()`.

Concrete failing path: select profile with a different `reference_path` than the currently loaded one; `_reference_digest` now describes the wrong file, and if the signal were blocked/ordered differently the saved `VoiceProfile.reference_sha256` no longer matches `reference_path`. The `test_gui_profile_fetch_…` assertion contradicts safe behavior here — the digest should be `""` after `_apply_service_voice`, not the old hash. This is a confirmed inconsistency between the stated guarantee ("stable roundtrip, permission reset") and what the test locks in.

### 2. `provider_fields` validation passes on the wrong schema level when voice-list items are `$ref`'d

`profile_contract()` builds `items` inline, so `resolve(voice_schema["items"]).get("properties")` works. But if a documented API uses `"items": {"$ref": "#/components/schemas/Voice"}` (the normal OpenAPI style — as the source's own `resolve()` exists to support), `resolve(voice_schema["items"])` returns the component with `properties`. That path is fine. However the *request-body* check earlier (`resolve(resolve(op["requestBody"])[...])`) assumes a `$ref` at the `requestBody` node. `voices_operation_id` GET path assumes `voice_op["responses"]["200"]` exists un-`$ref`'d. A documented GET returning via `"responses": {"$ref": ...}` or a `parameters`-driven 200 is rejected — not a false *accept*, so out of "confirmed bug" scope per your filter. **Skip — not load-bearing.**

### 3. Stale-fetch guard does not compare the *mapping file contents*, only its path string (confirmed)

Sequence:
1. Fetch starts with `settings.service_config = "<pathA>"`; `mapped_profiles()` runs in worker.
2. User *overwrites* `pathA` on disk with a different mapping (same path, new provider fields) — `service_config.text()` unchanged.
3. Worker returns profiles computed from the **old** contract; `ready()` sees `service_config.text() == settings.service_config` → accepts and installs them.
4. User selects one; `service_voice_id` comes from a contract that no longer matches the file that `configuration()`/`test_voice` will later load → provider ID rejected at synthesize time or, worse, silently sent to a different endpoint mapping.

The guard on `self.engine.currentData()` and the config-path string is insufficient because the contract identity (`ServiceContract.model_id` / sha256 of the OpenAPI) is not threaded to the GUI. The fix must compare a content identity (mtime+size or the contract hash), not the path string.

### 4. `_apply_service_voice` resets permission but not `_reference_digest`, and leaves a stale `language` index if profile language is `"en"/"sv"` mismatch — same root as #1

Consolidate: the permission reset is present and correct; the *digest* reset depends on `textChanged`, which only fires if `reference_path` actually changes. If the selected profile omits `reference_path` but changes `service_voice_id` (a supported case — reference WAV still required per your note), the digest stays tied to the previous engine, and `configuration()` produces a `VoiceProfile` whose `reference_sha256` was computed for a possibly different contract. Add `self._reference_digest = ""` **and** recompute on save, or leave it blank when `service_voice_id` changes under `documented-service`.

---

**Minimal proposal (GUI):** in `_apply_service_voice`, after setting `service_voice_id`, explicitly set `self._reference_digest = ""` (not relying on `textChanged`). In `fetch_service_profiles.ready`, capture `(service_config text, engine data, service_config mtime_ns, size)` at fetch time and re-check all four before installing; discard otherwise. Update `test_gui_profile_fetch_…` to assert `_reference_digest == ""` after profile selection — the current expectation conflicts with the digest-invalidation requirement.

**Not bugs (not restated):** no assumed endpoint/field names (validated against OpenAPI), worker-owned fetch, no implicit network, permission reset on profile select — all present and correct as supplied.
