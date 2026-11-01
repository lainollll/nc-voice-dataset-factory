"""Explicit OpenAPI-driven bridge. No assumed Chatterbox service endpoints."""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import soundfile as sf

from .audio import mono
from .domain import AudioResult, SynthesisSettings, VoiceProfile


@dataclass(frozen=True)
class ServiceContract:
    base_url: str
    path: str
    request_body: dict[str, Any]
    response_field: str | None
    model_id: str
    voices_path: str | None = None
    provider_fields: dict[str, str] | None = None

    @classmethod
    def load(cls, path: Path) -> ServiceContract:
        path = Path(path).resolve()
        if path.stat().st_size > 1_000_000:
            raise ValueError("Service configuration exceeds 1 MB")
        config = json.loads(path.read_text(encoding="utf-8"))
        url = urlsplit(config["base_url"])
        try:
            local = url.hostname == "localhost" or ipaddress.ip_address(url.hostname or "").is_loopback
        except ValueError:
            local = False
        if (
            url.scheme not in {"http", "https"}
            or not local
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Documented service must use a loopback local URL without embedded credentials")
        api = (path.parent / config["openapi_path"]).resolve()
        if api.stat().st_size > 10_000_000:
            raise ValueError("OpenAPI document exceeds 10 MB")
        raw = api.read_bytes()
        doc = json.loads(raw)
        if not str(doc.get("openapi", "")).startswith("3."):
            raise ValueError("A documented OpenAPI 3 JSON contract is required")

        def resolve(node):
            visited = set()
            while "$ref" in node:
                ref = node["$ref"]
                if not ref.startswith("#/") or ref in visited:
                    raise ValueError("Only non-circular internal OpenAPI references are supported")
                visited.add(ref)
                node = doc
                for key in ref[2:].split("/"):
                    node = node[key.replace("~1", "/").replace("~0", "~")]
            return node

        def operation(operation_id, method):
            matches = [
                (p, item[method])
                for p, item in doc.get("paths", {}).items()
                if method in item and item[method].get("operationId") == operation_id
            ]
            if len(matches) != 1:
                raise ValueError(f"Exactly one documented {method} operation is required: {operation_id}")
            route, op = matches[0]
            if (
                not route.startswith("/")
                or route.startswith("//")
                or any(c in route for c in "{}?#")
                or "://" in route
            ):
                raise ValueError("Service operation must have a literal documented path without parameters")
            return route, op

        route, op = operation(config["operation_id"], "post")
        body = config["request_body"]
        schema = resolve(resolve(op["requestBody"])["content"]["application/json"]["schema"])
        if (
            not isinstance(body, dict)
            or set(body) - set(schema.get("properties", {}))
            or set(schema.get("required", [])) - set(body)
        ):
            raise ValueError("Request fields must match the declared OpenAPI JSON schema")
        if "{text}" not in body.values():
            raise ValueError("The documented request must explicitly map a field to {text}")
        response = resolve(op["responses"]["200"])["content"]
        field = config.get("response_audio_field")
        if field:
            properties = resolve(response["application/json"]["schema"])["properties"]
            if field not in properties or resolve(properties[field]).get("type") != "string":
                raise ValueError("Base64 audio response field is not declared in the API")
        elif not ({"audio/wav", "audio/x-wav"} & response.keys()):
            raise ValueError("Documented response must be WAV audio or an explicitly mapped base64 field")
        voices_path = (
            operation(config["voices_operation_id"], "get")[0] if config.get("voices_operation_id") else None
        )
        provider_fields = config.get("provider_fields")
        if provider_fields is not None:
            if (
                not voices_path
                or not isinstance(provider_fields, dict)
                or not {"id", "name"} <= provider_fields.keys()
                or provider_fields.keys() - {"id", "name", "language", "reference_path"}
                or any(not isinstance(v, str) or not v.strip() for v in provider_fields.values())
            ):
                raise ValueError("Documented provider_fields requires id/name and a voice-list operation")
            _, voice_op = operation(config["voices_operation_id"], "get")
            voice_schema = resolve(
                resolve(voice_op["responses"]["200"])["content"]["application/json"]["schema"]
            )
            if voice_schema.get("type") != "array":
                raise ValueError("Documented voice-list response must be an array of profile objects")
            properties = resolve(voice_schema["items"]).get("properties", {})
            if set(provider_fields.values()) - set(properties):
                raise ValueError("Provider fields must be declared in the documented voice-list response")
        info = doc.get("info", {})
        identity = f"documented-service/{info.get('title')}/{info.get('version')}/sha256:{hashlib.sha256(raw).hexdigest()}"
        return cls(config["base_url"].rstrip("/"), route, body, field, identity, voices_path, provider_fields)


class DocumentedServiceTTS:
    def __init__(self, settings: SynthesisSettings) -> None:
        self.contract = ServiceContract.load(Path(settings.service_config))
        self.session = requests.Session()
        self.session.trust_env = False  # Local reference audio must never pass through an environment proxy.

    def synthesize(
        self, text: str, voice: VoiceProfile, settings: SynthesisSettings, seed: int
    ) -> AudioResult:
        voice.validate()
        start = time.perf_counter()
        values: dict[str, Any] = {
            **asdict(settings),
            "text": text,
            "language": voice.language,
            "voice_id": voice.service_voice_id or voice.id,
            "seed": seed,
            "reference_wav_base64": base64.b64encode(Path(voice.reference_path).read_bytes()).decode("ascii"),
        }

        def render(value):
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                key = value[1:-1]
                if key not in values:
                    raise ValueError(f"Unknown documented request placeholder: {key}")
                return values[key]
            if isinstance(value, dict):
                return {k: render(v) for k, v in value.items()}
            if isinstance(value, list):
                return [render(v) for v in value]
            return value

        response = self.session.post(
            self.contract.base_url + self.contract.path,
            json=render(self.contract.request_body),
            timeout=(5, 180),
            allow_redirects=False,
        )
        if response.is_redirect:
            raise ValueError("Service redirect refused; configure the documented local operation directly")
        response.raise_for_status()
        if len(response.content) > 100 * 1024**2:
            raise ValueError("Service audio response exceeds 100 MiB")
        raw = (
            base64.b64decode(response.json()[self.contract.response_field], validate=True)
            if self.contract.response_field
            else response.content
        )
        data, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        return AudioResult(
            mono(data),
            int(rate),
            self.contract.model_id,
            {"seed": seed, "operation": self.contract.path, "settings": asdict(settings)},
            time.perf_counter() - start,
        )

    def profiles(self) -> list[dict[str, Any]]:
        if not self.contract.voices_path:
            raise ValueError("Connected API does not expose a configured documented voice-list operation")
        response = self.session.get(
            self.contract.base_url + self.contract.voices_path, timeout=(5, 15), allow_redirects=False
        )
        if response.is_redirect:
            raise ValueError("Voice-list redirect refused")
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, list) or not all(isinstance(value, dict) for value in result):
            raise ValueError(
                "Voice-list operation must return JSON profile objects; configure a specific provider otherwise"
            )
        return result

    def close(self) -> None:
        self.session.close()

    def mapped_profiles(self) -> list[dict[str, str]]:
        fields = self.contract.provider_fields
        if not fields:
            raise ValueError("Configure documented provider_fields before selecting service voices")
        profiles = self.profiles()
        if len(profiles) > 10_000:
            raise ValueError("Service returned more than 10,000 profiles")
        result, seen = [], set()
        for raw in profiles:
            identifier, name = raw.get(fields["id"]), raw.get(fields["name"])
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, (str, int))
                or not str(identifier).strip()
                or len(str(identifier)) > 512
                or not isinstance(name, str)
                or not name.strip()
                or len(name) > 256
            ):
                raise ValueError("Service profile requires a nonempty stable id and name")
            identifier = str(identifier)
            if identifier in seen:
                raise ValueError("Service returned duplicate voice identifiers")
            seen.add(identifier)
            entry = {"service_voice_id": identifier, "name": name}
            for key in ("language", "reference_path"):
                value = raw.get(fields[key]) if key in fields else None
                if value is not None:
                    if not isinstance(value, str) or len(value) > 4096:
                        raise ValueError(f"Invalid service profile {key}")
                    entry[key] = value
            if entry.get("language") not in {None, "en", "sv"}:
                raise ValueError("Service voice language must be explicitly mapped to en or sv")
            if entry.get("reference_path") and not Path(entry["reference_path"]).is_absolute():
                raise ValueError("Service reference_path must be an absolute local WAV path")
            result.append(entry)
        return result
