from __future__ import annotations

import csv
import ipaddress
import json
import math
import random
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urlparse

import requests
from num2words import num2words

from .corpus import CATEGORIES, TEMPLATES, VALUES
from .domain import TextItem
from .storage import canonical_text

NORMALIZATION_VERSION = "nc-text-v2"
_ABBREVIATIONS = {
    "en": {"e.g.": "for example", "i.e.": "that is", "Dr.": "Doctor", "Mr.": "Mister"},
    "sv": {"t.ex.": "till exempel", "bl.a.": "bland annat", "dvs.": "det vill säga", "kl.": "klockan"},
}
_UNWANTED = re.compile(
    r"(?:ignore (?:all |previous )?instructions|system prompt|as an ai|<\|[^>]+\|>|```|^\s*(?:system|assistant|user)\s*:)",
    re.I | re.M,
)


def normalize_text(text: str, language: str = "en", expand_numbers: bool = True) -> str:
    if language not in VALUES:
        raise ValueError(f"Unsupported language: {language}")
    text = unicodedata.normalize("NFC", text).strip()
    if not text or any(unicodedata.category(ch) == "Cc" and not ch.isspace() for ch in text):
        raise ValueError("Empty text or control characters")
    if _UNWANTED.search(text):
        raise ValueError("Instructions, role markers or unwanted formatting in source text")
    for before, after in _ABBREVIATIONS[language].items():
        text = re.sub(r"(?<!\w)" + re.escape(before), after, text, flags=re.I)
    if expand_numbers:

        def convert(match: re.Match) -> str:
            raw = match.group()
            if len(raw.replace("-", "")) > 9:
                return " ".join(num2words(int(d), lang=language) for d in raw if d.isdigit())
            return num2words(int(raw), lang=language)

        # Do not guess ambiguous decimal/date/time conventions; preview retains those forms.
        text = re.sub(r"(?<![\w.,:/-])-?\d+(?![\d:/-]|[.,]\d)", convert, text)
    text = " ".join(text.split()).translate(
        str.maketrans({"“": '"', "”": '"', "’": "'", "—": ",", "–": "-", "…": ","})
    )
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


class DiversityIndex:
    """Exact keys plus an inverted shingle index; bounded candidate comparisons."""

    def __init__(self, threshold: float = 0.94) -> None:
        self.threshold = threshold
        self.keys: set[str] = set()
        self.tokens: list[tuple[str, ...]] = []
        self.postings: dict[tuple[str, ...], list[int]] = defaultdict(list)

    @staticmethod
    def _tokens(text: str) -> tuple[str, ...]:
        return tuple(
            w for w in canonical_text(text).split() if w not in {"a", "an", "the", "en", "ett", "den", "det"}
        )

    def add(self, text: str) -> bool:
        key = canonical_text(text)
        if not key or key in self.keys:
            return False
        tokens = self._tokens(text)
        shingles = {tokens[i : i + 3] for i in range(max(1, len(tokens) - 2))}
        candidates = Counter(index for shingle in shingles for index in self.postings[shingle][-200:])
        for index, _ in candidates.most_common(48):
            if SequenceMatcher(None, tokens, self.tokens[index], autojunk=False).ratio() >= self.threshold:
                return False
        index = len(self.tokens)
        self.tokens.append(tokens)
        self.keys.add(key)
        for shingle in shingles:
            self.postings[shingle].append(index)
        return True


def generate_texts(
    count: int,
    language: str = "en",
    weights: dict[str, float] | None = None,
    seed: int = 0,
    min_words: int = 6,
    max_words: int = 36,
    topics: list[str] | None = None,
    exclude: list[str] | None = None,
    index: DiversityIndex | None = None,
    assigned: Counter[str] | None = None,
) -> list[TextItem]:
    if language not in VALUES or not 0 <= count <= 100_000 or not 1 <= min_words <= max_words <= 200:
        raise ValueError("Invalid language, count or word limits")
    weights = weights or dict.fromkeys(CATEGORIES, 1.0)
    if (
        weights.keys() - set(CATEGORIES)
        or any(not math.isfinite(v) or v < 0 for v in weights.values())
        or not sum(weights.values())
    ):
        raise ValueError("Category weights must be finite, non-negative, and total more than zero")
    categories = [key for key in CATEGORIES if weights.get(key, 0) > 0]
    rng, index, result = random.Random(seed), index if index is not None else DiversityIndex(), []
    for old in exclude or []:
        index.add(old)
    assigned = assigned if assigned is not None else Counter()
    previous_count = sum(assigned.values())
    total_weight = sum(weights.values())
    for _ in range(max(1000, count * 100)):
        if len(result) >= count:
            break
        # Fill largest proportional deficit; category exhaustion is visible, never substituted.
        category = max(
            categories,
            key=lambda c: (previous_count + len(result) + 1) * weights[c] / total_weight - assigned[c],
        )
        template = rng.choice(TEMPLATES[language][category])
        values = {key: rng.choice(pool) for key, pool in VALUES[language].items()}
        values["number"] = str(rng.randint(2, 5000))
        if topics:
            values["topic"] = rng.choice(topics).strip()
        original = template.format(**values)
        spoken = normalize_text(original, language)
        if min_words <= len(spoken.split()) <= max_words and index.add(spoken):
            result.append(
                TextItem(
                    original,
                    spoken,
                    category,
                    language,
                    {
                        "method": "authored-template",
                        "seed": seed,
                        "template": template,
                        "normalization": NORMALIZATION_VERSION,
                    },
                )
            )
            assigned[category] += 1
    if len(result) < count:
        raise ValueError(
            f"Text corpus exhausted at {len(result)} unique sentences (limiting category: {category}); use imports or a local LLM"
        )
    return result


def diversity_statistics(items: list[TextItem]) -> dict[str, Any]:
    words = Counter(word for item in items for word in canonical_text(item.spoken_text).split())
    return {
        "sentences": len(items),
        "unique_words": len(words),
        "word_count": sum(words.values()),
        "type_token_ratio": len(words) / max(1, sum(words.values())),
        "categories": dict(Counter(item.category for item in items)),
        "character_coverage": "".join(sorted(set("".join(item.spoken_text.casefold() for item in items)))),
        "phonetic_coverage": "Character coverage proxy only; phoneme analysis has not run",
    }


def import_texts(
    path: Path, language: str = "en", text_field: str = "text"
) -> tuple[list[TextItem], list[str]]:
    path = Path(path)
    if path.stat().st_size > 100 * 1024**2:
        raise ValueError("Text import exceeds 100 MiB; split the file first")
    warnings, items, index = [], [], DiversityIndex()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".csv":
            source = csv.DictReader(handle)
        elif path.suffix.lower() == ".jsonl":

            def rows():
                for row, line in enumerate(handle, 1):
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        warnings.append(f"Line {row}: invalid JSON")

            source = rows()
        elif path.suffix.lower() == ".json":
            if path.stat().st_size > 10 * 1024**2:
                raise ValueError("JSON arrays are limited to 10 MiB; use streaming JSONL for larger imports")
            source = json.load(handle)
            if not isinstance(source, list):
                raise ValueError("JSON import must contain a list")
        elif path.suffix.lower() == ".txt":
            source = handle
        else:
            raise ValueError("Supported imports: TXT, CSV, JSON, JSONL")
        for row, entry in enumerate(source, 1):
            if row > 200_000:
                raise ValueError("Import exceeds 200,000 rows; split this dataset first")
            try:
                text = entry.get(text_field, "") if isinstance(entry, dict) else entry
                if not isinstance(text, str) or len(text) > 4000:
                    raise ValueError("text must be a string of at most 4,000 characters")
                spoken = normalize_text(text, language)
                if not index.add(spoken):
                    raise ValueError("duplicate or near-duplicate text")
                category = str(entry.get("category", "custom")) if isinstance(entry, dict) else "custom"
                items.append(
                    TextItem(
                        text.strip(),
                        spoken,
                        category,
                        language,
                        {
                            "method": "import",
                            "filename": path.name,
                            "row": row,
                            "normalization": NORMALIZATION_VERSION,
                        },
                    )
                )
            except ValueError as exc:
                warnings.append(f"Row {row}: {exc}")
    return items, warnings


def parse_llm_sentences(content: str, language: str) -> list[TextItem]:
    if _UNWANTED.search(content):
        raise ValueError("Local LLM returned unwanted formatting or instructions")
    try:
        values = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Local LLM must return a JSON list of sentence strings") from exc
    if not isinstance(values, list) or not 1 <= len(values) <= 100:
        raise ValueError("Expected 1–100 sentence strings")
    result, index = [], DiversityIndex()
    for value in values:
        if not isinstance(value, str) or len(value) > 2000 or "\n" in value:
            raise ValueError("Invalid sentence string")
        spoken = normalize_text(value, language)
        if index.add(spoken):
            result.append(TextItem(value, spoken, "custom", language, {"method": "local-llm"}))
    return result


def _validate_llm_endpoint(endpoint: str, allow_remote: bool) -> ParseResult:
    url = urlparse(endpoint)
    try:
        local = url.hostname == "localhost" or ipaddress.ip_address(url.hostname or "").is_loopback
    except ValueError:
        local = False
    if (
        url.scheme not in {"http", "https"}
        or url.username
        or url.password
        or (not local and not allow_remote)
    ):
        raise ValueError("Use a loopback OpenAI-compatible endpoint, or explicitly enable remote access")
    return url


def list_llm_models(endpoint: str, allow_remote: bool = False) -> list[str]:
    """Read OpenAI-compatible /models with the same endpoint rules as generation."""
    _validate_llm_endpoint(endpoint, allow_remote)
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(endpoint.rstrip("/") + "/models", timeout=(3, 10), allow_redirects=False)
    if response.is_redirect:
        raise ValueError("Model discovery does not follow redirects; use the server's actual base URL")
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError("The server returned an excessively large model list")
    try:
        data = response.json().get("data")
    except (ValueError, AttributeError) as exc:
        raise ValueError("The server did not return an OpenAI-compatible model list") from exc
    if not isinstance(data, list) or len(data) > 10_000:
        raise ValueError("The server did not return an OpenAI-compatible model list")
    models, seen = [], set()
    for row in data:
        model = row.get("id") if isinstance(row, dict) else None
        if (
            isinstance(model, str)
            and model.strip()
            and len(model) <= 512
            and not any(unicodedata.category(ch) == "Cc" for ch in model)
            and model not in seen
        ):
            models.append(model)
            seen.add(model)
    if not models:
        raise ValueError("No models were returned. Start or load a chat model in your server, then refresh.")
    return models


def local_llm_texts(
    endpoint: str,
    model: str,
    count: int,
    language: str,
    topic: str = "everyday conversation",
    allow_remote: bool = False,
    cancel: threading.Event | None = None,
) -> list[TextItem]:
    url = _validate_llm_endpoint(endpoint, allow_remote)
    if not model.strip() or not 1 <= count <= 100:
        raise ValueError("Specify a model and request 1–100 sentences")
    if cancel and cancel.is_set():
        raise InterruptedError("Text generation cancelled")
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            endpoint.rstrip("/") + "/chat/completions",
            json={
                "model": model,
                "temperature": 0.8,
                "max_tokens": min(8000, count * 100),
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only a JSON list of unique natural sentence strings. No Markdown, role labels, instructions or commentary.",
                    },
                    {
                        "role": "user",
                        "content": f"Write {count} distinct 6–36-word sentences in {'Swedish' if language == 'sv' else 'English'} about {topic[:500]}.",
                    },
                ],
            },
            timeout=(5, 90),
            allow_redirects=False,
        )
    finally:
        session.close()
    response.raise_for_status()
    if cancel and cancel.is_set():
        raise InterruptedError("Text generation cancelled")
    items = parse_llm_sentences(response.json()["choices"][0]["message"]["content"], language)
    return [
        TextItem(
            t.original_text,
            t.spoken_text,
            t.category,
            t.language,
            {**t.provenance, "model": model, "endpoint_host": url.hostname},
        )
        for t in items
    ]


def generate_llm_corpus(
    endpoint: str,
    model: str,
    count: int,
    language: str,
    min_words: int = 6,
    max_words: int = 36,
    topic: str = "everyday conversation",
    allow_remote: bool = False,
    cancel: threading.Event | None = None,
) -> list[TextItem]:
    if not 1 <= count <= 100_000:
        raise ValueError("Request 1–100,000 sentences")
    index, items = DiversityIndex(), []
    for _ in range(4 * math.ceil(count / 64)):
        if cancel and cancel.is_set():
            raise InterruptedError("Text generation cancelled")
        generated = local_llm_texts(
            endpoint, model, min(64, count - len(items)), language, topic, allow_remote, cancel
        )
        items.extend(
            item
            for item in generated
            if min_words <= len(item.spoken_text.split()) <= max_words and index.add(item.spoken_text)
        )
        if len(items) >= count:
            return items[:count]
    raise ValueError(
        f"Local LLM produced only {len(items)} unique sentences; not enough for the requested {count}"
    )


class ProjectTextSource:
    """One lazy text source per job. Imported sets never silently gain template text."""

    def __init__(self, config, repo, job_id: str | None = None) -> None:
        self.config, self.repo = config, repo
        self.sequence = repo.statistics()["total"]
        self.index = DiversityIndex()
        self.assigned: Counter[str] = Counter()
        for sample in repo.iter_samples():
            if sample["language"] == config.language:
                self.index.add(sample["spoken_text"])
                if job_id and sample["job_id"] == job_id:
                    self.assigned[sample["category"]] += 1

    def __call__(self, count: int) -> list[TextItem]:
        cfg = self.config
        if cfg.text_method == "imported":
            return []
        if cfg.text_method == "local-llm":
            items = local_llm_texts(
                cfg.llm_endpoint,
                cfg.llm_model,
                min(count, 64),
                cfg.language,
                ", ".join(cfg.custom_topics) or "everyday conversation",
                allow_remote=cfg.allow_remote_llm,
            )
            items = [
                t
                for t in items
                if cfg.min_words <= len(t.spoken_text.split()) <= cfg.max_words
                and self.index.add(t.spoken_text)
            ]
        else:
            items = generate_texts(
                count,
                cfg.language,
                cfg.category_weights,
                cfg.synthesis.seed + self.sequence,
                cfg.min_words,
                cfg.max_words,
                list(cfg.custom_topics),
                index=self.index,
                assigned=self.assigned,
            )
        self.sequence += count
        return items
