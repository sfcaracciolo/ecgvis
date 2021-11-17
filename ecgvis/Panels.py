from ecgvis.Views import ZarrTreeView
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QGroupBox, QLabel, QScrollArea, QVBoxLayout, QWidget

class TasksContainer(QScrollArea):
    def __init__(self, forms, parent: Optional[QWidget]) -> None:
        super().__init__(parent=parent)
        self.forms = forms
        self._actions = []
        self.setup_ui()

    def setup_ui(self):
        self.setMinimumWidth(500)

        # Tasks Container dock
        tasksWidget = QWidget()
        tasksLayout = QVBoxLayout()
        tasksWidget.setLayout(tasksLayout)

        #Scroll Area Properties
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setWidget(tasksWidget)

        for form, label in self.forms:
            tasksLayout.addWidget(form)
            form.setTitle(label)
            action = QAction(label)
            action.setCheckable(True)
            action.triggered.connect(form.toggle_visibility)
            self.parent().tasksMenu.addAction(action)
            # if label == 'Nonlinear Notch Adaptive Filter':
            #     action.trigger()
            self._actions.append(action)

        tasksLayout.addStretch(1)

class ZarrExplorer(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setup_ui()
        self.treeView.info_signal.connect(self.set_info)

    def setup_ui(self):
        self.setMinimumWidth(500)

        self.treeView = ZarrTreeView(parent=self)

        group_info = QGroupBox('Information')
        group_layout = QVBoxLayout()
        group_info.setLayout(group_layout)

        self.info = QLabel()
        self.info.setWordWrap(True)
        group_layout.addWidget(self.info)
        group_layout.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.treeView, stretch=1)
        layout.addWidget(group_info, stretch=1)
        self.setLayout(layout)

    def set_info(self, data):
        self.info.setText(data)