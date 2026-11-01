from __future__ import annotations

from pathlib import Path
from typing import Any

from .audio import sha256_file
from .storage import atomic_json, utcnow

REPOSITORIES = {
    "english": "ResembleAI/chatterbox",
    "multilingual-v2": "ResembleAI/chatterbox",
    "multilingual-v3": "ResembleAI/chatterbox",
    "asr": "Systran/faster-whisper-large-v3",
    "speaker": "speechbrain/spkrec-ecapa-voxceleb",
}
PATTERNS = {
    "english": ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"],
    "multilingual-v2": [
        "ve.pt",
        "t3_mtl23ls_v2.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
        "Cangjie5_TC.json",
    ],
    "multilingual-v3": [
        "ve.pt",
        "t3_mtl23ls_v3.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
        "Cangjie5_TC.json",
    ],
    "asr": ["model.bin", "config.json", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"],
    "speaker": ["hyperparams.yaml", "*.ckpt", "label_encoder.txt"],
}


def model_download_plan(
    kind: str, repository: str | None = None, revision: str | None = None
) -> dict[str, Any]:
    """Network access happens only after the user requests a download/inspection."""
    import fnmatch
    from huggingface_hub import HfApi

    if kind not in REPOSITORIES:
        raise ValueError("Unknown model type")
    repository = repository or REPOSITORIES[kind]
    info = HfApi().model_info(repository, revision=revision or "main", files_metadata=True)
    patterns = PATTERNS[kind]
    selected = [
        entry for entry in info.siblings if any(fnmatch.fnmatch(entry.rfilename, p) for p in patterns)
    ]
    if not selected:
        raise ValueError("No required files matched this model repository; inspect its documented layout")
    card = info.card_data.to_dict() if info.card_data else {}
    return {
        "kind": kind,
        "repository": repository,
        "revision": info.sha,
        "patterns": patterns,
        "files": [entry.rfilename for entry in selected],
        "bytes": sum(entry.size or 0 for entry in selected),
        "license": card.get("license", "See model card"),
        "source": f"https://huggingface.co/{repository}/tree/{info.sha}",
    }


def download_model(plan: dict[str, Any], destination: Path) -> str:
    """Explicit action; saves weights only to the selected new model folder."""
    from huggingface_hub import snapshot_download

    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Choose an empty model directory; existing weights are never replaced")
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        plan["repository"],
        revision=plan["revision"],
        allow_patterns=plan["patterns"],
        local_dir=destination,
        max_workers=2,
    )
    missing = [name for name in plan["files"] if not (destination / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete model download: {missing}")
    atomic_json(
        destination / "nc_model_identity.json",
        {
            **plan,
            "downloaded_at": utcnow(),
            "sha256": {name: sha256_file(destination / name) for name in plan["files"]},
        },
    )
    return str(destination)
