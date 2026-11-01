**No confirmed correctness defects found.**

Checked against the four requested axes:

1. **Signals/navigation** — All three signals (`navigate_requested`, `project_requested`, `small_test_requested`) are connected to existing slots (`open_page`, `choose_project`, `prepare_small_test`). The eight `step.page` strings ("Dashboard", "Voice Manager", "Text Generator", "Quality Control", "Batch Generator", "Dataset Browser", "Export") all match navigation titles created in `MainWindow`. Step 1 uses `action="project"` (default `page`), step 6 uses `"small-test"`; both dispatch correctly in `_activate`. Back/Next use `currentRowChanged`; boundary buttons are disabled and `_show_step` guards out-of-range (`-1` when Back sits on row 0 is filtered). Guide state persists across `open_page` re-entries because `GettingStartedGuide` is cached, and the test confirms step index survives.

2. **Unintended work/settings** — `_activate` only emits a signal; no `start_job`/`run_task` calls. `prepare_small_test` writes exactly `target_count`, `target_seconds`, `text_count`, then navigates. It does not touch `synthesis_spins` or call save/start. Busy guard on `runner`/`aux_runner` prevents target mutation mid-operation. No `repo`/`pipeline` construction.

3. **Instruction/control mismatch** — Step 6 text says "The button below only prepares the settings; click Start yourself… It will not change targets while another operation is active" — matches `prepare_small_test` behavior (`busy` early-return with status message). Step 1 button "Create / Open project…" matches `choose_project`. Step 7/8 mention revalidate/regenerate/resume, but the action is only "Open Dataset Browser"/"Open Export", consistent with the stated delegation-to-pages design.

4. **Lifecycle** — `GettingStartedGuide` owns no timers, threads, or file handles. `metrics_timer` belongs to `MainWindow`, not the guide. No dangling connections; signals are Qt-owned and cleaned on widget deletion.

One observation (not a defect): the guide widget itself calls nothing destructive; all mutations route through `MainWindow`, so it inherits MainWindow's lifecycle guarantees. No counterexample produced.
