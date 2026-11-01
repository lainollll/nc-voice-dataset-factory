"""Standalone entry point, copied to a run and executed only in its training venv."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


class Tee:
    def __init__(self, stream, log) -> None:
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


def execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__nc_training__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            if (Path.cwd() / "STOP").exists():
                raise InterruptedError("Training stopped by user")
            print(f"Executing training cell {index + 1}/{len(notebook['cells'])}", flush=True)
            exec(compile("".join(cell["source"]), f"{path.name}:cell-{index + 1}", "exec"), namespace)


def main() -> int:
    run_root = Path(__file__).resolve().parent
    os.chdir(run_root)
    status, code = "FAILED", 1
    with (run_root / "training.log").open("a", encoding="utf-8") as log:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
        try:
            notebook = run_root / "finetuning" / "notebooks" / "lfm3_audio_tts_sft_with_transformers.ipynb"
            execute_notebook(notebook)
            status, code = "COMPLETED", 0
        except InterruptedError as exc:
            print(str(exc), flush=True)
            status, code = "STOPPED", 130
        except Exception:
            traceback.print_exc()
        finally:
            sys.stdout, sys.stderr = original_out, original_err
    temporary = run_root / "result.tmp"
    temporary.write_text(json.dumps({"status": status, "exit_code": code}), encoding="utf-8")
    os.replace(temporary, run_root / "result.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
