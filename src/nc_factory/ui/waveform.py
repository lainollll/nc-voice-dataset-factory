from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..audio import read_audio


class WaveformWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: list[tuple[float, float]] = []
        self._duration: float = 0.0
        self.setMinimumHeight(48)

    def set_peaks(self, peaks: list[tuple[float, float]], duration: float) -> None:
        self._peaks = list(peaks)
        self._duration = float(duration)
        self.update()

    def clear(self) -> None:
        self._peaks = []
        self._duration = 0.0
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        try:
            rect = self.rect()
            painter.fillRect(rect, QColor("#11161d"))
            mid_y = rect.top() + rect.height() / 2.0
            half_h = rect.height() / 2.0 - 6.0
            painter.setPen(QPen(QColor("#283342"), 1))
            painter.drawLine(rect.left() + 6, int(mid_y), rect.right() - 6, int(mid_y))

            if self._peaks and rect.width() > 12 and half_h > 0:
                painter.setPen(QPen(QColor("#66d9ef"), 1))
                n = len(self._peaks)
                usable = rect.width() - 12
                for i, (lo, hi) in enumerate(self._peaks):
                    x = rect.left() + 6 + int(i * usable / n)
                    y1 = mid_y - max(-1.0, min(1.0, hi)) * half_h
                    y2 = mid_y - max(-1.0, min(1.0, lo)) * half_h
                    painter.drawLine(x, int(y1), x, int(y2))

            painter.setPen(QColor("#9aa4ad"))
            label = f"{self._duration:.2f}s" if self._duration > 0 else ""
            if label:
                painter.drawText(
                    QRectF(rect.left() + 8, rect.top() + 4, rect.width() - 16, 16),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )
        finally:
            painter.end()


def prepare_waveform(path: str, bins: int = 600) -> dict[str, object]:
    import numpy as np

    data, sample_rate = read_audio(path)
    samples = np.asarray(data, dtype=np.float64).reshape(-1)
    duration = len(samples) / sample_rate
    bins = max(1, int(bins))
    if samples.size == 0:
        return {"peaks": [], "duration": duration}

    edges = np.linspace(0, samples.size, bins + 1).astype(np.int64)
    peaks: list[tuple[float, float]] = []
    for i in range(bins):
        start = int(edges[i])
        stop = int(edges[i + 1])
        if stop <= start:
            if peaks:
                peaks.append(peaks[-1])
            else:
                peaks.append((0.0, 0.0))
            continue
        seg = samples[start:stop]
        peaks.append((float(seg.min()), float(seg.max())))

    return {"peaks": peaks, "duration": duration}
