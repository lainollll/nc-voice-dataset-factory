def test_window_starts(qtbot):
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert "NC Voice Dataset Factory" == window.windowTitle()
    assert window.navigation.count() >= 4


def test_task_runner_delivers_result_and_stays_owned(qtbot):
    from nc_factory.ui.workers import TaskRunner

    runner = TaskRunner()
    results = []
    runner.result.connect(results.append)
    with qtbot.waitSignal(runner.finished, timeout=5000):
        runner.start(lambda: 42)
        assert runner.busy
    assert results == [42]
    assert not runner.busy
