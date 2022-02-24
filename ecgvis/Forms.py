from PySide6.QtCore import QStringListModel, Qt
import numpy as np
from ecgvis.Viewers import *
from PySide6.QtGui import QColor, QDropEvent
from ecgvis.CustomWidgets import TaskForm, TensorWidget
from PySide6.QtWidgets import QColorDialog, QComboBox, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget
try:
    from ecg_tools.tools import nonlinear_adaptative_notch_filter, bdr_median_filter, isoline_correction, bdr_spline_filter
    from ecgi_tools.utils import laplace_interpolation
except ModuleNotFoundError:
    pass
# try:
#     import spdict
# except ModuleNotFoundError:
#     pass
# try:
#     import pywt
# except ModuleNotFoundError:
#     pass

MAX_OVERLAY = 5

class TemporalViewerForm(QGroupBox):
    def __init__(self, model, parent=None) -> None:
        super().__init__(parent=parent)

        self.model = model
        self._viewers = []
        self._default_colors = [ QColor(color) for color in ['white', 'red', 'green', 'blue', 'yellow'] ]
        self.setup_ui()
        self.setup_callbacks()

    def setup_callbacks(self):
        self.box_buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

    def toggle_visibility(self):
        if self.isHidden():
            self.show()
        else:
            self.hide()

    def setup_ui(self):
        
        self.grid_layout = QGridLayout()

        self.grid_layout.addWidget(QLabel('Tensor'), 0, 0)
        self.grid_layout.addWidget(QLabel('Scale'), 0, 1)
        self.grid_layout.addWidget(QLabel('Color'), 0, 2)

        self._rows = [ (index, TensorWidget(self.model), QSpinBox(), (QPushButton(), QColorDialog())) for index in range(MAX_OVERLAY) ]
        list(map(lambda r: self.set_row(r), self._rows))
        self.box_buttons = QDialogButtonBox(QDialogButtonBox.Apply)# | QDialogButtonBox.Cancel)
        
        layout = QVBoxLayout()
        layout.addLayout(self.grid_layout)
        layout.addWidget(self.box_buttons)
        layout.addStretch(1)
        self.setLayout(layout)
        self.hide()

    def set_row(self, row):
        i, input, scale, (button, picker) = row
        scale.setMinimum(1)
        scale.setMaximum(1000)
        scale.setValue(1)
        button.clicked.connect(picker.show)
        picker.setOption(QColorDialog.ShowAlphaChannel)
        picker.colorSelected.connect(lambda e, b=button: self.set_color(e, b))
        picker.setCurrentColor(self._default_colors[i])
        self.set_color(self._default_colors[i], button)

        self.grid_layout.addWidget(input, i+1, 0)
        self.grid_layout.addWidget(scale, i+1, 1)
        self.grid_layout.addWidget(button, i+1, 2)

    def set_color(self, qcolor, button):
        button.setStyleSheet(f'QPushButton {{background-color: {qcolor.name()}; color: {qcolor.name()};}}')

    def apply(self):
        inputs = self.get_inputs()
        if inputs[0]:
            self._viewers.append(TemporalViewer(self.model, *inputs))

    def get_inputs(self):
        tensors = []
        colors = []
        for _, tensor, scale, (_ , color) in self._rows:
            if tensor.path.text():
                tensor_data = tensor.get_tensor()
                tensor_data *= scale.value()
                tensors.append(tensor_data)
                colors.append(color.currentColor().getRgbF())

        return tensors, colors
        
class NonlinearNotchForm(TaskForm):
    def __init__(self, model, parent=None) -> None:
        super().__init__(model, parent=parent)
        self.model = model
        self.setup_ui()
        self.setup_callbacks()

    def setup_callbacks(self):
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

    def setup_ui(self):
        self.input_path = TensorWidget(self.model)
        self.input_alpha = QDoubleSpinBox()
        self.input_alpha.setMinimum(.1)
        self.input_alpha.setMaximum(100.)
        self.input_alpha.setSingleStep(.1)
        self.input_alpha.setValue(10.)
        self.input_f0 = QDoubleSpinBox()
        self.input_f0.setMinimum(.1)
        self.input_f0.setMaximum(10000.)
        self.input_f0.setSingleStep(.1)
        self.input_f0.setValue(50.)        
        self.input_fs = QSpinBox()
        self.input_fs.setMinimum(1)
        self.input_fs.setMaximum(2048)
        self.input_fs.setSingleStep(1)
        self.input_fs.setValue(1024)
        self.input_unit = QDoubleSpinBox()
        self.input_unit.setMinimum(.1)
        self.input_unit.setMaximum(100000.)
        self.input_unit.setSingleStep(.1)
        self.input_unit.setValue(1.) 

        self.form_layout.insertRow(0, 'Input', self.input_path)
        self.form_layout.insertRow(1, 'Rate', self.input_alpha)
        self.form_layout.insertRow(2, 'Frequency', self.input_f0)
        self.form_layout.insertRow(3, 'Sampling', self.input_fs)
        self.form_layout.insertRow(4, 'Unit', self.input_unit)

    def apply(self):

        y = nonlinear_adaptative_notch_filter(
            self.input_path.get_tensor(),
            alpha=self.input_alpha.value(),
            f0=self.input_f0.value(),
            fs=self.input_fs.value(),
            unit=self.input_unit.value()
        )

        self.save2zarr(y)
        
class MedianFilterForm(TaskForm):
    def __init__(self, model, parent=None) -> None:
        super().__init__(model, parent=parent)
        self.model = model
        self.setup_ui()
        self.setup_callbacks()

    def setup_callbacks(self):
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

    def apply(self):
        x = self.input.get_tensor()
        y = bdr_median_filter(x, self.window_size.value(), in_place=False)
        self.save2zarr(y)

    def setup_ui(self):

        self.setWindowTitle('BDR Median Filter')

        self.input = TensorWidget(self.model)
        self.window_size = QSpinBox()
        self.window_size.setMinimum(1)
        self.window_size.setMaximum(10000)
        self.window_size.setSingleStep(1)

        self.form_layout.insertRow(0, 'Input', self.input)
        self.form_layout.insertRow(1, 'Window size', self.window_size)

class IsolineCorrectionForm(TaskForm):
    def __init__(self, model, parent=None) -> None:
        super().__init__(model, parent=parent)
        self.model = model
        self.setup_ui()
        self.setup_callbacks()

    def setup_callbacks(self):
        self.engine.currentTextChanged.connect(lambda t: self.bins.setEnabled(True if t=='numpy' else False))
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

    def setup_ui(self):
        self.input = TensorWidget(self.model)
        self.start = QSpinBox()
        self.start.setMinimum(0)
        self.start.setMaximum(100)
        self.stop = QSpinBox()
        self.stop.setMinimum(0)
        self.stop.setMaximum(100)
        self.engine = QComboBox()
        self.engine.addItems(['scipy', 'numpy'])
        self.bins = QSpinBox()
        self.bins.setEnabled(False)
        self.bins.setMinimum(1)
        self.bins.setMaximum(1000)
        self.bins.setValue(100)

        self.form_layout.insertRow(0, 'Input', self.input)
        self.form_layout.insertRow(1, 'Start', self.start)
        self.form_layout.insertRow(2, 'Stop', self.stop)
        self.form_layout.insertRow(3, 'Engine', self.engine)
        self.form_layout.insertRow(4, 'Bins', self.bins)

    def apply(self):
        start = self.start.value()
        stop = self.stop.value() if self.stop.value() else None

        y = isoline_correction(
            self.input.get_tensor(),
            limits=(start, stop),
            engine=self.engine.currentText(),
            bins=self.bins.value(),
            in_place=False
        )
        
        self.save2zarr(y)

class SplineFilterForm(TaskForm):
    def __init__(self, model, parent=None) -> None:
        super().__init__(model, parent=parent)
        self.model = model
        self.setup_ui()
        self.setup_callbacks()

    def setup_callbacks(self):
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

    def setup_ui(self):
        self.input = TensorWidget(self.model)
        self.fiducials = TensorWidget(self.model)
        self.window_size = QSpinBox()
        self.window_size.setMinimum(1)
        self.window_size.setMaximum(100)
        self.order = QSpinBox()
        self.order.setMinimum(1)
        self.order.setMaximum(5)
        self.order.setValue(3)

        self.form_layout.insertRow(0, 'Input', self.input)
        self.form_layout.insertRow(1, 'Fiducials', self.fiducials)
        self.form_layout.insertRow(2, 'Window size', self.window_size)
        self.form_layout.insertRow(3, 'Order', self.order)

    def apply(self):

        y = bdr_spline_filter(
            self.input.get_tensor(),
            self.fiducials.get_tensor(),
            size=self.window_size.value(),
            order=self.order.value(),
            in_place=False
        )
        
        self.save2zarr(y)

class LaplaceInterpolationForm(TaskForm):
    def __init__(self, model, parent=None) -> None:
        super().__init__(model, parent=parent)
        self.model = model
        self.setup_ui()
        self.setup_callbacks()

    def setup_callbacks(self):
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

    def setup_ui(self):
        self.input_nodes = TensorWidget(self.model)
        self.input_faces = TensorWidget(self.model)
        self.input_values = TensorWidget(self.model)
        self.input_indices = TensorWidget(self.model)

        self.form_layout.insertRow(0, 'Nodes', self.input_nodes)
        self.form_layout.insertRow(1, 'Faces', self.input_faces)
        self.form_layout.insertRow(2, 'Values', self.input_values)
        self.form_layout.insertRow(3, 'Indices', self.input_indices)

    def apply(self):

        y = laplace_interpolation(
            self.input_nodes.get_tensor(),
            self.input_faces.get_tensor(),
            self.input_values.get_tensor(),
            self.input_indices.get_tensor().ravel(),
            in_place=False
        )
        
        self.save2zarr(y)

class MatrixViewerForm(QGroupBox):
    def __init__(self, model, parent=None) -> None:
        super().__init__(parent=parent)

        self.model = model
        self._viewers = []
        self.setup_ui()
        self.setup_callbacks()

    def setup_callbacks(self):
        self.box_buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

    def toggle_visibility(self):
        if self.isHidden():
            self.show()
        else:
            self.hide()

    def setup_ui(self):
        
        self.input = TensorWidget(self.model)
        self.box_buttons = QDialogButtonBox(QDialogButtonBox.Apply)# | QDialogButtonBox.Cancel)
        
        layout = QVBoxLayout()
        layout.addWidget(self.input)
        layout.addWidget(self.box_buttons)
        layout.addStretch(1)
        self.setLayout(layout)
        self.hide()

    def apply(self):
        if self.input.path.text():
            self._viewers.append(MatrixViewer(self.model, self.input.get_tensor()))
        else:
            self._viewers.append(MatrixViewer(self.model, np.zeros((1,1), dtype=np.int32)))

class SpatioTemporalViewerForm(QGroupBox):
    def __init__(self, model, parent=None) -> None:
        super().__init__(parent=parent)

        self.model = model
        self._viewers = []
        self.setup_ui()
        self.setup_callbacks()

        # self.set_test_tensors()

    def set_test_tensors(self):
        self.toggle_visibility()
        self.nodes.path.setText('Meshes/Torso/electrodes')
        self.nodes.sliding.setText(':,:')
        self.faces.path.setText('Meshes/Torso/faces')
        self.faces.sliding.setText(':,:')
        self.values.path.setText('Interventions/dog2_beat1_SR/Torso/Signals/interpolated')
        self.values.sliding.setText(':,:')
        self.values.transpose.setChecked(True)

    def setup_callbacks(self):
        self.box_buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

    def toggle_visibility(self):
        if self.isHidden():
            self.show()
        else:
            self.hide()

    def setup_ui(self):
        
        self.nodes = TensorWidget(self.model)
        self.faces = TensorWidget(self.model)
        self.values = TensorWidget(self.model)
        self.box_buttons = QDialogButtonBox(QDialogButtonBox.Apply)# | QDialogButtonBox.Cancel)
        
        layout = QFormLayout()
        layout.addRow('Nodes', self.nodes)
        layout.addRow('Faces', self.faces)
        layout.addRow('Values', self.values)
        layout.addWidget(self.box_buttons)
        # layout.addStretch(1)
        self.setLayout(layout)
        self.hide()

    def apply(self):
        conds = [
            self.nodes.path.text(),
            self.faces.path.text(),
            self.values.path.text()
        ]
        if all(conds):
            self._viewers.append(SpatioTemporalViewer(self.model, self.nodes.get_tensor(), self.faces.get_tensor(), self.values.get_tensor()))

# class SparseDictBuilderForm(TaskForm):
    
#     def __init__(self, model, parent=None) -> None:
#         super().__init__(model, parent=parent)
#         self.model = model
#         self.field_widgets = []
#         self.label_widgets = []
#         self.setup_ui()
#         self.setup_callbacks()
#         self.set_dict_type(0)

#     def setup_callbacks(self):
#         self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
#         self.input_type.currentIndexChanged.connect(lambda i: self.set_dict_type(i))
        
#     def setup_ui(self):
#         # self.form_layout.setVerticalSpacing(0)

#         self.input_samples = QSpinBox()
#         self.input_samples.setMinimum(1)
#         self.input_samples.setMaximum(2048)
#         self.input_samples.setSingleStep(1)

#         self.input_step = QSpinBox()
#         self.input_step.setMinimum(1)
#         self.input_step.setMaximum(2048)
#         self.input_step.setSingleStep(1)

#         self.input_type = QComboBox()
#         self.input_type.addItems(['Identity', 'Stationary Wavelet Transform', 'Wavelet Packet', 'Continuous Wavelet Transform'])

#         self.form_layout.insertRow(0,'Samples', self.input_samples)
#         self.form_layout.insertRow(1,'Step', self.input_step)
#         self.form_layout.insertRow(2,'Type', self.input_type)

#         self.id_ui()
#         self.swt_ui()
#         self.wpd_ui()
#         self.cwt_ui()
        
#         self.insertWidgets()

#     def id_ui(self):

#         # identiy
#         # self._continuous_families = pywt.families(short=False)[7:10] # only continuous non complex wavelets
#         # self._short_continuous_families = pywt.families(short=True)[7:10] # only continuous non complex wavelets
#         self.field_widgets.append([QWidget()])
#         self.label_widgets.append([QLabel()])

#     def swt_ui(self):
#         # swt
#         level = QSpinBox()
#         level.setMinimum(1)
#         level.setMaximum(2048)
#         level.setSingleStep(1)

#         family = QComboBox()
#         model = QStringListModel()
#         model.setStringList(pywt.families(short=True)[:7])
#         family.setModel(model)

#         order = QComboBox()
#         model = QStringListModel()
#         self.setDiscreteOrder(model, 0)
#         order.setModel(model)
        
#         family.currentIndexChanged.connect(lambda t: self.setDiscreteOrder(model, t))

#         self.field_widgets.append([level, family, order])
#         self.label_widgets.append(['Max. Level','Wavelet Family','Wavelet Order'])

#     def wpd_ui(self):
#         # swt
#         level = QSpinBox()
#         level.setMinimum(1)
#         level.setMaximum(2048)
#         level.setSingleStep(1)

#         family = QComboBox()
#         model = QStringListModel()
#         model.setStringList(pywt.families(short=True)[:7])
#         family.setModel(model)

#         order = QComboBox()
#         model = QStringListModel()
#         self.setDiscreteOrder(model, 0)
#         order.setModel(model)

#         family.currentIndexChanged.connect(lambda t: self.setDiscreteOrder(model, t))

#         self.field_widgets.append([level, family, order])
#         self.label_widgets.append(['Max. Level','Wavelet Family','Wavelet Order'])

#     def cwt_ui(self):
#         # swt
#         level = QSpinBox()
#         level.setMinimum(1)
#         level.setMaximum(2048)
#         level.setSingleStep(1)

#         family = QComboBox()
#         model = QStringListModel()
#         model.setStringList(pywt.families(short=True)[7:10])
#         family.setModel(model)

#         order = QComboBox()
#         model = QStringListModel()
#         self.setContinuousOrder(model, 0)
#         order.setModel(model)

#         family.currentIndexChanged.connect(lambda t: self.setContinuousOrder(model, t))

#         self.field_widgets.append([level, family, order])
#         self.label_widgets.append(['Max. Level','Wavelet Family','Wavelet Order'])

#     def insertWidgets(self):
#         i = 3
#         for names, widgets in zip(self.label_widgets, self.field_widgets):
#             for n, w in zip(names, widgets):
#                 self.form_layout.insertRow(i, n, w)
#                 i += 1

#     def set_dict_type(self, ix):
#         self.hideRows()
#         self.setVisibleRows(self.field_widgets[ix], visible=True)

#     def setVisibleRows(self, widgets, visible=True):
#         for w in widgets:
#             row, role = self.form_layout.getWidgetPosition(w)
            
#             field = self.form_layout.itemAt(row, role).widget()
#             field.setVisible(visible)

#             label = self.form_layout.labelForField(field)
#             label.setVisible(visible)
            
#     def hideRows(self):
#         for widgets in self.field_widgets:
#             self.setVisibleRows(widgets, visible=False)
        
#     def setDiscreteOrder(self, model, e):
#         f = pywt.families(short=True)[:7][e]
#         model.setStringList(pywt.wavelist(family=f))

#     def setContinuousOrder(self,model, e):
#         f = pywt.families(short=True)[7:10][e]
#         model.setStringList(pywt.wavelist(family=f))

#     def apply(self):
#         ix = self.input_type.currentIndex()
#         samples = self.input_samples.value()
#         step = self.input_step.value()

#         fields = self.field_widgets[ix]

#         if  ix == 0:
#             atoms = [np.ones(1, dtype=np.float32)]
#         elif ix == 1: # swt
#             max_level = fields[0].value()
#             wname = fields[2].currentText()
#             atoms = spdict.utils.get_swt_atoms(wname, max_level)
#         elif ix == 2: # wpd
#             max_level = fields[0].value()
#             wname = fields[2].currentText()
#             atoms = spdict.utils.get_wp_atoms(wname, max_level)
#         elif ix == 3: # cwt
#             max_level = fields[0].value()
#             wname = fields[2].currentText()
#             atoms = spdict.utils.get_cwt_atoms(wname, max_level)

#         D = spdict.utils.create_sparse_dict_from_atoms(atoms, samples, step)
#         self.save2zarr(D.toarray())

class FPTForm(QGroupBox):
    def __init__(self, model, parent=None) -> None:
        super().__init__(parent=parent)
        self.model = model
        self._list_viewers = []
        self.setup_ui()
        self.setup_callbacks()

        self.set_default()

    def setup_callbacks(self):
        self.box_buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)

    def toggle_visibility(self):
        if self.isHidden():
            self.show()
        else:
            self.hide()

    def setup_ui(self):
        self.signal = TensorWidget(self.model)
        self.fpt = TensorWidget(self.model)
        self.bad_beats = TensorWidget(self.model)
        self.input_window = QSpinBox()
        self.input_window.setMinimum(-1)
        self.input_window.setMaximum(2000)
        self.input_window.setValue(0)
        self.box_buttons = QDialogButtonBox(QDialogButtonBox.Apply)# | QDialogButtonBox.Cancel)

        layout = QFormLayout()
        layout.addRow('ECG', self.signal)
        layout.addRow('FPT', self.fpt)
        layout.addRow('Bad beats', self.bad_beats)
        layout.addRow('Window size', self.input_window)
        layout.addWidget(self.box_buttons)
        # layout.addStretch(1)
        self.setLayout(layout)
        self.hide()

    def set_default(self):
        self.signal.path.setText('164/raw_bp_2_300_pli_iso')
        self.signal.sliding.setText(':')
        self.fpt.path.setText('164/fpt')
        self.fpt.sliding.setText(':,:')
        self.bad_beats.path.setText('164/bad_beats')
        self.bad_beats.sliding.setText(':')
        self.input_window.setValue(330)

        self.show()

    def apply(self):
        v = FPTViewer(
            self.model,
            self.signal.get_tensor(),
            self.fpt.get_tensor(),
            self.bad_beats.get_tensor(),
            window = self.input_window.value(),
            parent=self.parent()
        )
        self._list_viewers.append(v)