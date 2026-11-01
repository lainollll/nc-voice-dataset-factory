from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_smoke(output: Path, language: str = "en", count: int = 10) -> dict:
    """Model-free plumbing test. Tones and source-backed mock ASR are clearly labeled."""
    import numpy as np
    from .audio import sha256_file, write_wav_atomic
    from .domain import ProjectConfig, QualitySettings, SynthesisSettings, VoiceProfile
    from .export import ExportOptions, export_dataset, validate_export
    from .pipeline import Pipeline
    from .quality import MockASR, QualityController
    from .storage import Repository, atomic_json
    from .text import generate_texts
    from .tts import MockTTS

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Smoke output must be a new directory")
    output.mkdir(parents=True)
    reference = output / "test_reference_NOT_A_VOICE.wav"
    t = np.arange(48000) / 24000
    write_wav_atomic(reference, (0.1 * np.sin(2 * np.pi * 170 * t)).astype("float32"), 24000)
    voice = VoiceProfile(
        "mock-reference",
        "MOCK REFERENCE — NOT A VOICE",
        str(reference),
        sha256_file(reference),
        language,
        True,
        "test-tone",
    )
    config = ProjectConfig(
        "MOCK SMOKE — NOT SPEECH",
        voice,
        language,
        synthesis=SynthesisSettings(engine="mock", device="cpu"),
        quality=QualitySettings(verify_asr=True, min_duration=1),
        target_count=count,
        max_attempts=max(count * 3, 10),
        minimum_free_gb=0,
    )
    repo = Repository(output / "project")
    repo.save_config(config)
    job = repo.create_job(config, generate_texts(count, language, seed=11))
    asr = MockASR(lambda path: repo.get_sample(path.stem.split("_")[0])["training_text"])
    result = Pipeline(repo, tts=MockTTS(), quality=QualityController(config.quality, asr=asr)).run(job)
    if result["accepted"] != count:
        raise RuntimeError(f"Mock smoke did not reach target: {result}")
    report = export_dataset(repo, output / "NC_Voice_Dataset", ExportOptions())
    validation = validate_export(Path(report.destination))
    result.update(export=report.destination, verification="MOCK ONLY", integrity=validation)
    atomic_json(output / "smoke_report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="NC Voice Dataset Factory")
    parser.add_argument("--project", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--smoke", type=Path, metavar="NEW_OUTPUT_FOLDER")
    actions.add_argument("--diagnostics", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--validate", type=Path, metavar="EXPORTED_DATASET")
    actions.add_argument("--run-config", type=Path, metavar="PROJECT_CONFIG_JSON")
    actions.add_argument("--resume-job", metavar="JOB_ID")
    actions.add_argument("--control", choices=["pause", "resume", "stop"])
    parser.add_argument("--job-id")
    parser.add_argument("--export-to", type=Path)
    actions.add_argument(
        "--download-model", choices=["english", "multilingual-v2", "multilingual-v3", "asr", "speaker"]
    )
    parser.add_argument("--language", choices=["en", "sv"], default="en")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--model-repo")
    args = parser.parse_args()
    try:
        if args.status:
            if not args.project:
                parser.error("--status requires --project")
            from .control import read_project_status

            print(json.dumps(read_project_status(args.project), ensure_ascii=False, indent=2))
            return 0
        if args.control:
            if not args.project or not args.job_id:
                parser.error("--control requires --project and --job-id")
            from .control import request_job_control

            request_job_control(args.project, args.job_id, args.control)
            print(f"Queued {args.control} for {args.job_id}")
            return 0
        if args.smoke:
            print(json.dumps(run_smoke(args.smoke, args.language, args.count), indent=2, ensure_ascii=False))
            return 0
        if args.diagnostics:
            from .diagnostics import diagnostics

            print(json.dumps(diagnostics(args.project), indent=2, ensure_ascii=False))
            return 0
        if args.validate:
            from .export import validate_export

            result = validate_export(args.validate)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["valid"] else 1
        if args.run_config or args.resume_job:
            if not args.project:
                parser.error("--run-config requires --project")
            from .domain import ProjectConfig
            from .storage import Repository
            from .pipeline import Pipeline
            from .text import ProjectTextSource

            repo = Repository(args.project)
            from .logging_config import configure_logging

            configure_logging(repo.root)
            if args.resume_job:
                job = args.resume_job
                config = ProjectConfig.from_dict(json.loads(repo.get_job(job)["config_json"]))
            else:
                config = ProjectConfig.from_dict(json.loads(args.run_config.read_text(encoding="utf-8")))
                config.validate()
                repo.save_config(config)
                job = repo.create_job(config, [])
            source = ProjectTextSource(config, repo, job)
            pipeline = Pipeline(
                repo,
                text_source=source,
                on_progress=lambda event: print(json.dumps(event, ensure_ascii=False)),
            )
            try:
                result = pipeline.run(job)
            except KeyboardInterrupt:
                pipeline.request_stop()
                return 130
            if result["status"] == "COMPLETED" and args.export_to:
                from .export import export_dataset

                print(export_dataset(repo, args.export_to))
            return 0 if result["status"] == "COMPLETED" else 1
        if args.download_model:
            if not args.destination:
                parser.error("--download-model requires --destination")
            from .models import download_model, model_download_plan

            plan = model_download_plan(args.download_model, args.model_repo, args.revision)
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            print(download_model(plan, args.destination))
            return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    from PySide6.QtWidgets import QApplication
    from .ui.window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(args.project)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
