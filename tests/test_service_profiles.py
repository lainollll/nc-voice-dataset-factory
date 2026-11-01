from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from test_service import contract_files


def profile_contract(tmp_path, url):
    mapping = contract_files(tmp_path, url)
    api = tmp_path / "openapi.json"
    doc = json.loads(api.read_text())
    doc["paths"]["/documented-voices"] = {
        "get": {
            "operationId": "listPermittedProfiles",
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "provider_key": {"type": "string"},
                                        "display": {"type": "string"},
                                        "locale": {"type": "string"},
                                        "wav": {"type": "string"},
                                    },
                                },
                            }
                        }
                    }
                }
            },
        }
    }
    api.write_text(json.dumps(doc))
    cfg = json.loads(mapping.read_text())
    cfg.update(
        voices_operation_id="listPermittedProfiles",
        provider_fields={
            "id": "provider_key",
            "name": "display",
            "language": "locale",
            "reference_path": "wav",
        },
    )
    mapping.write_text(json.dumps(cfg))
    return mapping


def test_explicit_documented_profile_mapping_and_no_implicit_fetch(tmp_path, config):
    from nc_factory.service import DocumentedServiceTTS

    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    [
                        {
                            "provider_key": "provider-42",
                            "display": "Permitted voice",
                            "locale": "sv",
                            "wav": config.voice.reference_path,
                        }
                    ]
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        path = profile_contract(tmp_path, f"http://127.0.0.1:{server.server_port}")
        backend = DocumentedServiceTTS(replace(config.synthesis, service_config=str(path)))
        try:
            assert not observed
            profiles = backend.mapped_profiles()
            assert observed == ["/documented-voices"]
            assert profiles == [
                {
                    "service_voice_id": "provider-42",
                    "name": "Permitted voice",
                    "language": "sv",
                    "reference_path": config.voice.reference_path,
                }
            ]
        finally:
            backend.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_profile_mapping_rejects_undocumented_fields_before_network(tmp_path):
    from nc_factory.service import ServiceContract

    path = profile_contract(tmp_path, "http://127.0.0.1:9999")
    cfg = json.loads(path.read_text())
    cfg["provider_fields"]["id"] = "invented_key"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="documented|declared"):
        ServiceContract.load(path)


def test_gui_profile_fetch_is_worker_owned_and_id_round_trips(tmp_path, config, qtbot, monkeypatch):
    from nc_factory.storage import Repository
    from nc_factory.ui.window import MainWindow
    from nc_factory.ui import voice_page

    observed = []
    main_thread = threading.get_ident()

    class Backend:
        def mapped_profiles(self):
            observed.append(threading.get_ident())
            return [{"name": "Service voice", "service_voice_id": "stable-provider-id", "language": "sv"}]

        def close(self):
            observed.append("closed")

    monkeypatch.setattr(voice_page, "create_tts", lambda settings: Backend())
    window = MainWindow()
    qtbot.addWidget(window)
    window.repo = Repository(tmp_path / "project")
    window.apply_config(config)
    window.engine.setCurrentIndex(window.engine.findData("documented-service"))
    window.service_config.setText(str(tmp_path / "explicit-mapping.json"))
    assert not observed
    window.fetch_service_profiles()
    qtbot.waitUntil(lambda: not window.runner.busy, timeout=5000)
    assert observed[0] != main_thread and observed[1] == "closed"
    window.service_voices.setCurrentIndex(1)
    assert window.voice_name.text() == "Service voice"
    assert not window.permission.isChecked()
    assert window._reference_digest == config.voice.reference_sha256
    selected = window.configuration()
    assert selected.voice.service_voice_id == "stable-provider-id"
    assert selected.voice.source == "documented-service"
    window.apply_config(selected)
    assert window.configuration().voice.service_voice_id == "stable-provider-id"
    window.service_config.setText(str(tmp_path / "another-mapping.json"))
    assert not window.service_voice_id.text()
