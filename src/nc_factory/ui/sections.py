"""Small expandable sections for optional controls; no application state."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setAccessibleName(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle.setObjectName("sectionToggle")
        self.setObjectName("expandableSection")
        layout.addWidget(self.toggle)
        self.content = QWidget()
        self.content.setObjectName("expandableBody")
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(self.content)
        self.content.hide()
        self.toggle.toggled.connect(self.setExpanded)

    def isExpanded(self) -> bool:
        return self.toggle.isChecked()

    @Slot(bool)
    def setExpanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.content.setVisible(expanded)
