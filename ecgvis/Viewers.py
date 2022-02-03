import vispy as vp
from vispy import visuals
from vispy.scene.widgets import widget
from ecgvis.Views import FPTView, MatrixTableView
from typing import Optional
from ecgvis.CustomWidgets import CursorsWidget, LineEditWithDrop
from ecgvis.Models import CursorsModel, FPTModel, MatrixModel
from ecgvis.Constants import *
from vispy.visuals.transforms.linear import STTransform
from ecgvis.CustomVisuals import Isolines, LinePicking, Lines, MarkersPicking, RegionsLine
from PySide6.QtCore import QSize, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout
import numpy as np
from superqt import QRangeSlider
from vispy import scene
import pathlib
from datetime import datetime
import ecg_tools
import fpt_tools

class SpatioTemporalViewer(QDialog):
    def __init__(self, model, vertices, faces, values, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.model = model

        self.n_samples = values.shape[1]
        self.min_value, self.max_value = values.min().round(decimals=2), values.max().round(decimals=2)

        self.meshdata = vp.geometry.MeshData(
            vertices=vertices,
            faces=faces,
            )

        self.values = values
        self.previous_idx = 0
        self.series = np.empty((self.n_samples, 2), dtype=np.float32)
        self.series[:, 0] = np.arange(self.n_samples)

        self.setup_canvas()
        self.setup_ui()
        self.setup_callbacks()

        self.set_time(0)
        self.set_marker(0)

        self.vb1.camera.set_range()
        self.vb2.camera.set_range(y=self.mesh.clim)

        # self.set_range()

    def setup_callbacks(self):
        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_mouse_press)

    def set_time(self, value):
        self.meshdata.set_vertex_values(self.values[:,value])
        self.mesh.mesh_data_changed()

    def set_series(self, idx):
        self.series[:,1] = self.values[idx,:]
        self.line.set_data(pos=self.series)

    def on_mouse_release(self, e):
        v = self.canvas.visual_at(e.pos)

        if v == self.vb1:
            idx = self.scatter.pick_marker(self.canvas, e)
            if idx is not None:
                self.set_marker(idx)

        self.line.drop_cursor()

    def set_marker(self, idx):
        self.scatter.set_color(self.previous_idx)
        self.scatter.set_color(idx, color=RED)
        self.set_series(idx)
        self.previous_idx = idx

    def on_key_press(self, e):
        key = e.key.name

        if key == 'Escape':
            self.close()

    def setup_ui(self):
        self.setWindowTitle('SpatioTemporal Viewer')

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas.native)

        self.setLayout(vlayout)
        self.show()

    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor='black', parent=self)
        # self.canvas.measure_fps()
        grid = self.canvas.central_widget.add_grid()

        self.vb1 = grid.add_view(
            row=0,
            col=0,
            border_color='white',
            border_width=0.,
            camera = 'arcball'

        )
        # scene.visuals.XYZAxis(parent=self.vb1.scene)

        cmap = vp.color.get_colormap('turbo')
        clim = (self.min_value, self.max_value)

        cbar = self.setup_colorbar(cmap, clim)
        self.mesh = self.setup_mesh(cmap, clim)
        self.scatter = self.setup_markers()

        self.vb1.add(self.mesh)
        self.vb1.add(self.scatter)

        wcbar = grid.add_widget(
            widget=cbar,
            row=0,
            col=1,
        )
        wcbar.width_max = 60

        self.vb2 = grid.add_view(
            row=1,
            col=0,
            col_span=2,
            border_color='white',
            border_width=0.,
            camera = 'panzoom'

        )
        self.vb2.height_max = 150
        self.vb2.camera.interactive = False

        self.line = self.setup_line()
        self.vb2.add(self.line)

    def setup_markers(self):
        scatter = MarkersPicking(
            self.meshdata.get_vertices(),
            WHITE,
            parent=self.vb1.scene,
            size=7,
            click_radius=5
        )
        return scatter

    def on_mouse_move(self, e):
        pos = self.line.move_cursor(self.canvas, e)
        if pos is not None:
            self.set_time(pos)

    def on_mouse_press(self, e):
        self.line.pick_cursor(self.canvas, e)

    def setup_line(self):
        line = LinePicking(
            parent=self.vb2.scene
        )
        return line

    def setup_colorbar(self, cmap, clim):
        scene.widgets.colorbar.ColorBarVisual.text_padding_factor = 2.5
        colorbar = scene.widgets.ColorBarWidget(
            cmap,
            'left',
            label_color='white',
            axis_ratio=0.05,
            clim = clim,
        )

        return colorbar

    def setup_mesh(self, cmap, clim):

        mesh = scene.visuals.Mesh(
            shading='flat',
            meshdata = self.meshdata,
        )
        wireframe_filter = vp.visuals.filters.WireframeFilter(width=.1, color=WHITE)

        mesh.cmap = cmap
        mesh.clim = clim
        mesh.attach(wireframe_filter)
        
        return mesh

class TemporalViewer(QDialog):

    def __init__(self, model, matrices, colors, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        # super().__init__(flags=Qt.WindowStaysOnTopHint, parent=parent)
        self.model = model
        self._list_matrices = matrices
        self.n_overlay = len(matrices)
        self.n_channels = max(map(lambda m: m.shape[1], matrices)) # tomo el mayor (deberian ser todos iguales), Lleno con 0 los faltantes.
        self.n_series =  min([self.n_channels, 10])
        self.series_size = max(map(lambda m: m.shape[0], matrices))
        self.next_sample = 0

        # self._list_pos_offset = []
        self._list_colors = map(lambda a: np.array(a, dtype=np.float32), colors) # colors # ['white', 'yellow', 'red', 'green', 'blue']
        self._list_scrolling_lines = []
        
        self.y_min = min(map(lambda m: m.min(), matrices))
        self.y_max = max(map(lambda m: m.max(), matrices))

        self.offset_series = np.maximum(np.abs(self.y_max), np.abs(self.y_min))

        self.x_range = [0, self.series_size]
        self.y_range = [-(1-0.5)*self.offset_series, 0.5*self.offset_series]
        self.y_lim = [-self.n_channels*self.offset_series, self.offset_series]

        self.setup_canvas()
        self.setup_ui()
        self.setup_callbacks()
        self.set_range()

        self.x_zoom_factor = 1.1
        self.y_zoom_factor = 1.1
        
        self.selected_cursor = None
        # self.canvas.measure_fps()

    def setup_callbacks(self):
        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_mouse_wheel)
        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_mouse_double_click)
        self.canvas.connect(self.on_mouse_press)
        self.hslider.valueChanged.connect(self.hslider_change)
        self.cursors_widget.load_button.clicked.connect(self.load)

    def hslider_change(self, value):
        value = self.hslider.value() # tuve que agregar esto porque value = 0 siempre. No se por que 
        self.x_range = value
        self.set_range()

    def setup_ui(self):
        self.setWindowTitle('Temporal Viewer')

        self.hslider = QRangeSlider(orientation=Qt.Horizontal, parent=self)
        self.hslider.setMinimum(0)
        self.hslider.setMaximum(self.series_size)
        self.hslider.setValue((0, self.series_size))
        
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas.native)
        vlayout.addWidget(self.hslider)

        self.cursors_model = CursorsModel()
        self.cursors_widget = CursorsWidget(self.model)
        proxy_model = QSortFilterProxyModel()
        proxy_model.sort(0)
        proxy_model.setSourceModel(self.cursors_model)
        self.cursors_widget.cursors_view.setModel(proxy_model)

        hlayout = QHBoxLayout()
        hlayout.addLayout(vlayout, stretch=5)
        hlayout.addWidget(self.cursors_widget, stretch=1)

        self.setLayout(hlayout)
        self.show()

    def setup_panel(self):

        self.text_view = self.grid.add_view(
            row=0,
            col=0,
            border_color='white',
            border_width=0.
        )

        self.text_view.width_max = 60
        self.text_view.camera = 'panzoom'
        self.text_view.camera.interactive = False

        for i in range(self.n_channels):
            scene.visuals.Text(
                text=f'{i}',
                pos=(0, -i*self.offset_series),
                color='white',
                font_size=9.,
                parent=self.text_view.scene
            )

        ch_markers = np.empty((self.n_channels, 2), dtype=np.float32)
        ch_markers[:, 0] = 5
        ch_markers[:, 1] = np.arange(self.n_channels)
        ch_markers[:, 1] *= -self.offset_series

        scene.visuals.Markers(
            pos=ch_markers,
            symbol='arrow',
            size=10,
            face_color='white',
            parent=self.text_view.scene
        )

    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor='black', parent=self)
        # self.canvas.measure_fps()
        self.grid = self.canvas.central_widget.add_grid()
        self.setup_panel()
        self.setup_plot()

    def setup_plot(self):

        self.view = self.grid.add_view(
            row=0,
            col=1,
            border_color='white',
            border_width=0.
        )
        self.view.interactive = False
        self.view.camera = 'panzoom'
        self.view.camera.interactive = False

        Isolines(
            n_lines=self.n_channels,
            line_size=self.series_size,
            dx=1.,
            color=LIGHT_AMBER,
            offset=self.offset_series,
            parent=self.view.scene,
        )

        for matrix, color in zip(self._list_matrices, self._list_colors):
            size, n_series = matrix.shape
            lines = Lines(   
                n_lines=n_series,
                line_size=size,
                dx=1.,
                color=color,
                offset=self.offset_series,
                parent=self.view.scene,
            )

            lines.transform = STTransform()
            self._list_scrolling_lines.append(lines)
            for i in range(n_series):
                lines.set_data(i, matrix[:,i])

    def set_range(self):
        # print(self.x_range)
        self.view.camera.set_range(x=self.x_range, y=self.y_range, margin=0.)
        self.text_view.camera.set_range(x=(-10, 10), y=self.y_range, margin=0.)

    def add_channel(self):
        y_min = self.y_range[0] - self.offset_series
        if y_min > self.y_lim[0]:
            self.y_range[0] -= self.offset_series
            self.set_range()

    def remove_channel(self):
        y_min = self.y_range[0] + self.offset_series
        if y_min < self.y_range[1]:
            self.y_range[0] += self.offset_series
            self.set_range()
        
    def scale(self, factor):
        for lines in self._list_scrolling_lines:
            lines.set_scale(factor)

    def on_key_press(self, e):
        key = e.key.name # e.native.text()
        modifiers = e.modifiers

        if len(modifiers) > 0: # zoom en x.
            if modifiers[0].name == 'Shift' and key == '+':
                self.scale(self.y_zoom_factor)
            elif modifiers[0].name == 'Shift' and key == '-':
                self.scale(1/self.y_zoom_factor)
            return
            
        if key == '+':
            self.add_channel()

        if key == '-':
            self.remove_channel()

        if key == 'S':
            self.screenshot()

        if key == 'Escape':
            self.close()

    def on_mouse_press(self, e):
        # c = self.canvas.visual_at(e.pos)
        cs = self.canvas.visuals_at(e.pos, radius=10)
        if len(cs):
            c = cs[0]
            if c in self.cursors_model._data:
                self.pick_cursor(c)

    def on_mouse_double_click(self, e):
        tr = self.canvas.scene.node_transform(self._list_scrolling_lines[0])
        pos = tr.map(e.pos)[0] 
        self.add_cursor(pos)

    def on_mouse_move(self, e):
        if self.selected_cursor is not None:
            if e.is_dragging:
                tr = self.canvas.scene.node_transform(self.selected_cursor)
                dx = tr.map(e.pos)[0] #  - tr.map(e.last_event.pos)[0]
                self.selected_cursor.transform.move(dx)
                self.cursors_model.layoutChanged.emit() # seria mejor dataChanged

    def on_mouse_release(self, e):
        self.drop_cursor()

    def on_mouse_wheel(self, e):
        step = e.delta[1]*self.offset_series
        y_range = [self.y_range[0]+step, self.y_range[1]+step]
        if (y_range[0] > self.y_lim[0]) and (y_range[1] < self.y_lim[1]):
            self.y_range[0] += step
            self.y_range[1] += step
            self.set_range()

    def add_cursor(self, pos):
        cursor = scene.visuals.InfiniteLine(
            pos=0.,
            color=AMBER,
            vertical=True,
            parent=self.view.scene
        )
        cursor.interactive = True
        cursor.transform = STTransform(translate=pos)
        self.cursors_model._add_cursor(cursor)

    def pick_cursor(self, cursor):
        self.drop_cursor()
        cursor.set_data(color=RED)
        cursor.update()
        self.selected_cursor = cursor

    def drop_cursor(self):
        if self.selected_cursor is not None:
            self.selected_cursor.set_data(color=AMBER)
            self.selected_cursor.update()
            self.selected_cursor = None

    def clear(self):
        for cursor in self.cursors_model._data:
            self.view.scene._remove_child(cursor)
        self.view.scene.update()
        self.cursors_model.clear()

    def add_cursors(self, cursors):
        self.clear()
        for pos in cursors:
            self.add_cursor(pos)

    def load(self):
        arr = self.cursors_widget.input_tensor.get_tensor()
        self.add_cursors(arr.tolist())

    def screenshot(self):
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img = self.screen().grabWindow(self.winId())
        img.save(f'screenshots/{dt}.png', 'png', quality=100)
        
class MatrixViewer(QDialog):
    def __init__(self, model, matrix, parent=None) -> None:
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.model = model
        self.matrix_model = MatrixModel(matrix)

        self.setup_ui()
        self.setup_callbacks()

    def setup_ui(self):
        self.setWindowTitle('Matrix Viewer')

        self.table_view = MatrixTableView(self.matrix_model)

        self.output = LineEditWithDrop()
        self.save_button = QPushButton('Save')
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.output, stretch=5)
        save_layout.addWidget(self.save_button, stretch=1)

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addLayout(save_layout)
        self.setLayout(layout)
        self.show()

    def setup_callbacks(self):
        self.save_button.clicked.connect(self.save2zarr)

    def save2zarr(self, sparse=False):
        array = self.matrix_model._data
        path = self.output.text()
        # if sparse
        try:
            self.model.zarr_root.create_dataset(path, data=array)
        except:
            print('Dataset already exists.')

class FPTViewer(QDialog):

    def __init__(self, model, signal, fpt, bad_beats, window, parent=None) -> None:
        super().__init__(parent=parent)
        self.model = model
        self.signal = np.ravel(signal)

        self.centered_fpt = fpt_tools.tools.centered_fpt(fpt, window)
        self.centered_fpt[ self.centered_fpt == fpt_tools.tools.nan_value(fpt)] =  -1

        self.table_data = fpt
        self.table_data[ self.table_data == fpt_tools.tools.nan_value(fpt)] =  -1

        table_mask = np.ones(fpt.shape[0], dtype=np.bool8)
        table_mask[bad_beats] = False

        self.window_size = window
        self.table_model = FPTModel(self.table_data, table_mask)

        # self.matrix, self.onsets, _ = ecg_tools.utils.sliding_window_from_centers(
        self.matrix, _, _ = ecg_tools.utils.sliding_window_from_centers(
            self.signal, 
            np.ravel(fpt[:,5]), 
            window
        )

        self.setup_canvas()
        self.setup_ui()
        self.setup_callbacks()

    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor='black', size=(800,200), parent=self)
        # self.canvas.measure_fps()
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.interactive = True
        self.view.camera.set_range(
            x=(0, self.window_size-1),
            y=(1.1*self.signal.min(), 1.1*self.signal.max()),
            margin=0.
        )

        self.line = RegionsLine(self.window_size, parent=self.view.scene)

    def setup_ui(self):
        self.setWindowTitle('FPT Viewer')
        self.table = FPTView(self.table_model)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.table, stretch=3)
        vlayout.addWidget(self.canvas.native, stretch=1)
        self.setLayout(vlayout)
        self.resize(QSize(800, 1000))
        self.show()

    def setup_callbacks(self):
        self.table.row.connect(self.set_line)

    def set_line(self, index):

        points = np.array(
            [
                [self.centered_fpt[index, 0], self.centered_fpt[index, 2]], # Pon, Poff
                [self.centered_fpt[index, 3], self.centered_fpt[index, 7]], # QRSon, QRSoff
                [self.centered_fpt[index, 9], self.centered_fpt[index, 11]], # Ton, Toff
            ], dtype = np.float32
        )
        # points = np.where(points == -1, -1, points-self.onsets[index])
        # points[np.any(points == -1, axis=1),:] = -1
        
        markers = np.array(
            [
                [self.centered_fpt[index, 1], 0], # Ppos, Ppeak
                [self.window_size//2, 0],
                # [self.table_data[index, 5], 0], # Rpos, Rpeak
                [self.centered_fpt[index, 6], 0], # Spos, Speak
                [self.centered_fpt[index, 10], 0], # Tpos, Tpeak
            ], dtype = np.float32
        )
        
        # print(index, markers[:,0], self.onsets[index], markers[:,0]-self.onsets[index])
        # markers[:,0] = np.where(markers[:,0] == -1, -1, markers[:,0]-self.onsets[index])
        markers[:,1] = np.where(markers[:,0] == -1, 0, self.matrix[index, markers[:,0].astype(np.int32)])

        self.line.set_data(
            self.matrix[index, :],
            points,
            markers
        )