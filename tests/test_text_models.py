from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def llm_server():
    state = {
        "paths": [],
        "posts": [],
        "routes": {},
        "started": threading.Event(),
        "release": threading.Event(),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["paths"].append(self.path)
            if self.path.startswith("/old/"):
                state["started"].set()
                state["release"].wait(5)
            status, payload, headers = state["routes"].get(
                self.path,
                (
                    200,
                    {
                        "data": [
                            {"id": "chat-alpha"},
                            {"id": "chat-beta"},
                            {"id": "chat-alpha"},
                            {"other": "ignore"},
                        ]
                    },
                    {},
                ),
            )
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_POST(self):
            state["posts"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            content = json.dumps(
                [
                    "Please bring your blue umbrella when you leave tomorrow.",
                    "The museum beside the river opens early every morning.",
                ]
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        state["release"].set()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_lists_openai_compatible_model_ids_and_rejects_nonlocal_endpoint(llm_server):
    from nc_factory.text import list_llm_models

    base, state = llm_server
    assert list_llm_models(base + "/v1/") == ["chat-alpha", "chat-beta"]
    assert state["paths"] == ["/v1/models"]
    with pytest.raises(ValueError, match="loopback|remote"):
        list_llm_models("http://192.0.2.1:1234/v1")


@pytest.mark.parametrize("payload", [{"data": []}, {"wrong": []}, {"data": [{"id": "\n"}]}])
def test_discovery_rejects_empty_or_invalid_model_list(llm_server, payload):
    from nc_factory.text import list_llm_models

    base, state = llm_server
    state["routes"]["/v1/models"] = (200, payload, {})
    with pytest.raises(ValueError, match="model|Model"):
        list_llm_models(base + "/v1")


def test_model_discovery_does_not_follow_redirects(llm_server):
    from nc_factory.text import list_llm_models

    base, state = llm_server
    state["routes"]["/v1/models"] = (302, {}, {"Location": base + "/elsewhere"})
    with pytest.raises(ValueError, match="redirect"):
        list_llm_models(base + "/v1")
    assert state["paths"] == ["/v1/models"]


def test_dropdown_auto_populates_preserves_custom_id_and_drives_existing_preview(
    qtbot, tmp_path, config, llm_server
):
    from nc_factory.storage import Repository
    from nc_factory.ui.window import MainWindow

    base, state = llm_server
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_page("Text Generator")
    window.llm_endpoint.setText(base + "/v1")
    assert state["paths"] == []  # Built-in corpus needs no server.
    window.llm_model.setEditText("saved-custom-id")
    window.text_method.setCurrentIndex(window.text_method.findData("local-llm"))
    qtbot.waitUntil(lambda: window.llm_model.count() == 2, timeout=5000)
    assert window.llm_model.currentText() == "saved-custom-id"
    window.llm_model.setCurrentIndex(window.llm_model.findText("chat-beta"))
    window.repo = Repository(tmp_path / "project")
    window.apply_config(
        replace(config, text_method="local-llm", llm_endpoint=base + "/v1", llm_model="chat-beta")
    )
    assert window.configuration().llm_model == "chat-beta"
    window.text_count.setValue(2)
    qtbot.waitUntil(lambda: not window.aux_runner.busy)
    window.generate_preview()
    qtbot.waitUntil(lambda: not window.runner.busy, timeout=5000)
    assert len(window._preview_items) == 2
    assert state["posts"][0]["model"] == "chat-beta"
    assert window.repo.list_jobs() == [] and window.pipeline is None


def test_stale_discovery_cannot_fill_dropdown_for_another_endpoint(qtbot, llm_server):
    from PySide6.QtCore import QTimer
    from nc_factory.ui.window import MainWindow

    base, state = llm_server
    state["routes"]["/old/v1/models"] = (200, {"data": [{"id": "old-server-model"}]}, {})
    state["routes"]["/new/v1/models"] = (200, {"data": [{"id": "new-server-model"}]}, {})
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_page("Text Generator")
    window.llm_endpoint.setText(base + "/old/v1")
    window.text_method.setCurrentIndex(window.text_method.findData("local-llm"))
    try:
        qtbot.waitUntil(state["started"].is_set, timeout=5000)
        tick = []
        QTimer.singleShot(0, lambda: tick.append(True))
        qtbot.waitUntil(lambda: bool(tick))  # HTTP is blocked; the UI remains responsive.
        window.llm_endpoint.setText(base + "/new/v1")
    finally:
        state["release"].set()
        qtbot.waitUntil(lambda: not window.aux_runner.busy, timeout=5000)
    qtbot.waitUntil(lambda: window.llm_model.currentText() == "new-server-model", timeout=5000)
    assert window.llm_model.findText("old-server-model") == -1


def test_source_controls_retain_values_and_model_errors_allow_retry(qtbot, llm_server):
    from nc_factory.ui.window import MainWindow

    base, state = llm_server
    state["routes"]["/v1/models"] = (503, {}, {})
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_page("Text Generator")
    assert window.builtin_options.isVisible() and not window.llm_options.isVisible()
    window.min_words.setValue(8)
    window.llm_endpoint.setText(base + "/v1")
    window.llm_model.setEditText("custom-id")
    window.text_method.setCurrentIndex(window.text_method.findData("local-llm"))
    qtbot.waitUntil(lambda: "Unable" in window.llm_models_status.text(), timeout=5000)
    assert window.llm_model.currentText() == "custom-id"
    assert window.llm_options.isVisible() and not window.builtin_options.isVisible()
    qtbot.waitUntil(lambda: not window.aux_runner.busy)
    state["routes"].clear()
    window.refresh_models_button.click()
    qtbot.waitUntil(lambda: window.llm_model.count() == 2, timeout=5000)
    qtbot.waitUntil(lambda: not window.aux_runner.busy)
    window.text_method.setCurrentIndex(window.text_method.findData("imported"))
    assert not window.min_words.isEnabled() and not window.text_count.isEnabled()
    window.text_method.setCurrentIndex(window.text_method.findData("builtin"))
    assert window.min_words.isEnabled() and window.min_words.value() == 8


def test_closing_before_discovery_debounce_prevents_hidden_http(qtbot, llm_server):
    from nc_factory.ui.window import MainWindow

    base, state = llm_server
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_page("Text Generator")
    window.llm_endpoint.setText(base + "/v1")
    window.text_method.setCurrentIndex(window.text_method.findData("local-llm"))
    window.close()
    qtbot.wait(650)
    try:
        assert state["paths"] == []
    finally:
        qtbot.waitUntil(lambda: not window.aux_runner.busy, timeout=5000)


def test_text_model_controls_fit_compact_window(qtbot, llm_server):
    from nc_factory.ui.window import MainWindow

    base, _state = llm_server
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1000, 700)
    window.show()
    window.open_page("Text Generator")
    window.llm_endpoint.setText(base + "/v1")
    window.text_method.setCurrentIndex(window.text_method.findData("local-llm"))
    qtbot.waitUntil(lambda: window.llm_model.count() == 2, timeout=5000)
    qtbot.waitUntil(lambda: not window.aux_runner.busy)
    assert window.pages.currentWidget().horizontalScrollBar().maximum() == 0
