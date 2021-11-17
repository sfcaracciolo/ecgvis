import pathlib
from typing import Optional
from PySide6.QtCore import QRegularExpression, Qt, Signal
from PySide6.QtGui import QDropEvent, QRegularExpressionValidator
from PySide6.QtWidgets import QCheckBox, QDialogButtonBox, QFormLayout, QGroupBox, QLineEdit, QListView, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
import numpy as np
import zarr


class CursorsWidget(QWidget):
    def __init__(self, model, parent: Optional[QWidget]=None) -> None:
        super().__init__(parent=parent)

        self.model = model
        self.cursors_view = QListView()
        self.cursors_view.setEnabled(False)

        layout = QVBoxLayout()

        self.input_tensor = TensorWidget(model)
        self.load_button = QPushButton()
        self.load_button.setText('Load')
        self.input_tensor.layout().addWidget(self.load_button)
        input_group = QGroupBox('Input')
        input_group.setLayout(self.input_tensor.layout())

        self.output = LineEditWithDrop()
        self.sort_checkbox = QCheckBox()
        self.sort_checkbox.setChecked(True)
        self.round_checkbox = QCheckBox()
        self.round_checkbox.setChecked(True)
        self.save_button = QPushButton()
        self.save_button.setText('Save')

        output_layout = QFormLayout()
        input_path_layout = QHBoxLayout()
        input_path_layout.addWidget(self.output)
        input_path_layout.addWidget(self.save_button)
        output_layout.addRow('Path', input_path_layout)
        output_layout.addRow('Sort', self.sort_checkbox)
        output_layout.addRow('Round', self.round_checkbox)
        output_group = QGroupBox('Output')
        output_group.setLayout(output_layout)
        
        layout.addWidget(input_group)
        layout.addWidget(self.cursors_view)
        layout.addWidget(output_group)

        self.setLayout(layout)

        self.save_button.clicked.connect(self.save2zarr)

    def save2zarr(self, sparse=False):

        array = self.cursors_view.model().sourceModel().get_values()
        if self.round_checkbox.checkState() == Qt.Checked:
            array = np.rint(array).astype(np.int32)
        if self.sort_checkbox.checkState() == Qt.Checked:
            array.sort()

        path = self.output.text()
        # if sparse
        try:
            self.model.zarr_root.create_dataset(path, data=array)
        except:
            print('Dataset already exists.')

class TensorWidget(QWidget):
    def __init__(self, model, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent=parent)
        self.model = model
        self.path = LineEditWithDrop()
        self.sliding = LineEditSliding()
        self.transpose = QCheckBox()

        layout = QHBoxLayout()
        layout.addWidget(self.path, stretch=4)
        layout.addWidget(self.sliding, stretch=2)
        layout.addWidget(self.transpose)

        self.setLayout(layout)

    def get_tensor(self):
        arr = self.model.zarr_root[self.path.text()]
        narr = eval(f'arr.oindex[{self.sliding.text()}]')
        narr = narr[:,np.newaxis] if narr.ndim == 1 else narr
        narr = narr.T if self.transpose.isChecked() else narr
        return narr

class LineEditSliding(QLineEdit):
    def __init__(self):
        super().__init__(parent=None)
        rx = QRegularExpression(r'(((\d*:?\d*)|(\[(\d+,?)+\])),)+') # https://regexper.com/
        validator = QRegularExpressionValidator(rx)
        self.setValidator(validator)

class LineEditWithDrop(QLineEdit):

    drop_signal = Signal(str, object)

    def __init__(self):
        super().__init__(parent=None)
        self.setAcceptDrops(True)

    def dropEvent(self, a0: QDropEvent) -> None:
        self.clear()
        super().dropEvent(a0)
        self.drop_signal.emit(a0.mimeData().text(), self)

class TaskForm(QGroupBox):
    def __init__(self, model, parent=None) -> None:

        super().__init__(parent=parent)
        
        self.model = model
        self.form_layout = QFormLayout()
        self.output = LineEditWithDrop()
        self.form_layout.addRow('Output', self.output)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply) #  | QDialogButtonBox.Cancel)
        
        layout = QVBoxLayout()
        layout.addLayout(self.form_layout)
        layout.addWidget(self.buttons)

        layout.addStretch(1)
        self.setLayout(layout)

        self.hide()

    def toggle_visibility(self):
        if self.isHidden():
            self.show()
        else:
            self.hide()

    def save2zarr(self, array, sparse=False):
        path = self.output.text()
        # if sparse
        try:
            self.model.zarr_root.create_dataset(path, data=array)
        except:
            print('Dataset already exists.')
