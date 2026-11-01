from __future__ import annotations

import io
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf


def contract_files(tmp_path, url):
    doc = {
        "openapi": "3.0.3",
        "info": {"title": "Test documented API", "version": "1"},
        "paths": {
            "/documented-synthesis": {
                "post": {
                    "operationId": "speak",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["prompt"],
                                    "properties": {
                                        "prompt": {"type": "string"},
                                        "reference": {"type": "string"},
                                    },
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}
                    },
                }
            }
        },
    }
    api = tmp_path / "openapi.json"
    api.write_text(json.dumps(doc), encoding="utf-8")
    cfg = tmp_path / "service.json"
    cfg.write_text(
        json.dumps(
            dict(
                openapi_path="openapi.json",
                base_url=url,
                operation_id="speak",
                request_body={"prompt": "{text}", "reference": "{reference_wav_base64}"},
            )
        ),
        encoding="utf-8",
    )
    return cfg


def test_documented_local_service_synthesizes_without_assumed_endpoint(tmp_path, config):
    from nc_factory.tts import create_tts

    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            observed.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            buffer = io.BytesIO()
            sf.write(buffer, np.zeros(24000, dtype="float32"), 24000, format="WAV", subtype="PCM_16")
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.end_headers()
            self.wfile.write(buffer.getvalue())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cfg = contract_files(tmp_path, f"http://127.0.0.1:{server.server_port}")
        settings = replace(config.synthesis, engine="documented-service", service_config=str(cfg))
        backend = create_tts(settings)
        result = backend.synthesize("Read this exact sentence.", config.voice, settings, 42)
        assert result.sample_rate == 24000 and len(result.samples) == 24000
        assert observed[0][0] == "/documented-synthesis"
        assert observed[0][1]["prompt"] == "Read this exact sentence."
        assert observed[0][1]["reference"]
        assert "Test documented API" in result.model_id
        backend.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_service_rejects_undocumented_fields_or_nonlocal_host(tmp_path):
    from nc_factory.service import ServiceContract

    cfg = contract_files(tmp_path, "https://example.com")
    with pytest.raises(ValueError, match="local|loopback"):
        ServiceContract.load(cfg)
    cfg = contract_files(tmp_path, "http://127.0.0.1:9999")
    data = json.loads(cfg.read_text())
    data["request_body"]["invented_field"] = "anything"
    cfg.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="declared|document"):
        ServiceContract.load(cfg)
