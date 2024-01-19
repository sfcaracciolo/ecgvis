import vispy as vp
from vispy import visuals
from vispy.scene.widgets import widget
from ecgvis.Views import FPTView, MatrixTableView
from ecgvis.CustomWidgets import CursorsWidget, LineEditWithDrop
from ecgvis.Models import CursorsModel, FPTModel, MatrixModel
from ecgvis.Constants import *
from vispy.visuals.transforms.linear import STTransform
from ecgvis.CustomVisuals import Isolines, LinePicking, Lines, MarkersPicking, RegionsLine
from PySide6.QtCore import QSize, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import QSlider, QDialog, QHBoxLayout, QPushButton, QVBoxLayout
import numpy as np
import scipy as sp
from superqt import QRangeSlider, QLabeledDoubleSlider, QLabeledSlider
from vispy import scene
from datetime import datetime
import ecg_tools
import fpt_tools
from collections import defaultdict
import open3d as o3d
class AlphaShapesViewer(QDialog):
    def __init__(self, model, nodes, values, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

        self.model = model
        self.nodes = nodes
        self.values = values

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(nodes)
        self.tetra_mesh, self.pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(self.pcd) # convex hull

        self.tetra = None
        
        self.setup_canvas()
        self.setup_ui()
        self.setup_callbacks()
        self.vb1.camera.set_range()
        self.set_alpha(values[0])

    def setup_callbacks(self):
        self.canvas.connect(self.on_key_press)
        self.hslider.valueChanged.connect(self.set_alpha)

    def setup_ui(self):
        self.setWindowTitle('Alpha Shapes Viewer')

        self.hslider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        self.hslider.setRange(self.values[0], self.values[-1])
        
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas.native)
        vlayout.addWidget(self.hslider)

        self.setLayout(vlayout)
        self.show()
    
    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor=BG_COLOR, parent=self)
        # self.canvas.measure_fps()
        grid = self.canvas.central_widget.add_grid()

        self.vb1 = grid.add_view(
            row=0,
            col=0,
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
            camera = 'arcball'

        )

        self.meshdata = vp.geometry.MeshData(
            vertices=self.nodes,
            faces=np.array([[0,1,2]], dtype=np.int32)
        )

        self.scatter = scene.visuals.Markers(
            pos=self.nodes,
            # pos=np.zeros((1,3), dtype=np.float32),
            size=10,
            face_color=RED,
            edge_width=0,
        )

        self.mesh = scene.visuals.Mesh(
            # shading='flat',
            meshdata = self.meshdata,
        )
        wireframe_filter = vp.visuals.filters.WireframeFilter(width=.75, color=BLACK)
        self.mesh.attach(wireframe_filter)
        
        self.vb1.add(self.mesh)
        self.vb1.add(self.scatter)

    def set_alpha(self, value):
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.pcd, value, self.tetra_mesh, self.pt_map)
        except IndexError:
            print('ERROR')
        else:
        # self.vertices, self.edges, self.faces = self.alpha_shape_3D(self.nodes, value)
        # if self.faces.shape[0] > 0:
            mesh.orient_triangles()
            self.faces = np.asarray(mesh.triangles)
            self.vertices = np.asarray(mesh.vertices)
            # if self.has_outliers():
            #     self.outliers = np.setd
            # self.scatter.set_data(self.outliers)
            self.meshdata.set_vertices(self.vertices)
            self.meshdata.set_faces(self.faces)
            self.meshdata.set_face_colors(np.broadcast_to(WHITE, (self.faces.shape[0], 4)))
            self.mesh.mesh_data_changed()
            # print(f'Has outliers: {self.has_outliers()} Is closed: {self.is_closed()}')

    def on_key_press(self, e):
        key = e.key.name

        if key == 'Escape':
            self.close()

    def has_outliers(self): 
        return self.nodes.shape[0] - self.vertices.size

    def is_closed(self):
        return self.vertices.size - self.edges.shape[0] + self.faces.shape[0] == 2
    
    def alpha_shape_3D(self, pos, alpha):
        """
        Compute the alpha shape (concave hull) of a set of 3D points.
        Parameters:
            pos - np.array of shape (n,3) points.
            alpha - alpha value.
        return
            outer surface vertex indices, edge indices, and triangle indices
        """
        if self.tetra is None:
            self.tetra = sp.spatial.Delaunay(pos)
            # Find radius of the circumsphere.
            # By definition, radius of the sphere fitting inside the tetrahedral needs 
            # to be smaller than alpha value
            # http://mathworld.wolfram.com/Circumsphere.html
            tetrapos = np.take(pos,self.tetra.vertices,axis=0)
            normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
            ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
            a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
            Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
            Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
            Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
            c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
            self.r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

        # Find tetrahedrals
        tetras = self.tetra.vertices[self.r<alpha,:]
        # triangles
        TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
        Triangles = tetras[:,TriComb].reshape(-1,3)
        Triangles = np.sort(Triangles,axis=1)
        
        # Remove triangles that occurs twice, because they are within shapes
        TrianglesDict = defaultdict(int)
        for tri in Triangles:
            TrianglesDict[tuple(tri)] += 1
            Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
        # #edges
        EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
        Edges=Triangles[:,EdgeComb].reshape(-1,2)
        Edges=np.sort(Edges,axis=1)
        Edges=np.unique(Edges,axis=0)

        Vertices = np.unique(Edges)
        return Vertices,Edges,Triangles
        # return None, None ,Triangles

class IsolinesViewer(QDialog):
    def __init__(self, model, vertices, faces, values, levels, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

        self.model = model
        self.n_samples = values.shape[1]
        self.vertices = vertices
        self.faces = faces

        self.meshdata = vp.geometry.MeshData(
            vertices=vertices,
            faces=faces,
        )


        self.values = values.flatten()
        # self.levels = levels.flatten()
        distinc_values = np.unique(self.values)
        n_values = distinc_values.size
        n_levels = 5
        self.levels = distinc_values[::round(n_values/n_levels)]
        print(distinc_values)
        print(self.levels)
        print(10*"=")
        self.previous_idx = 0
        self.series = np.empty((self.n_samples, 2), dtype=np.float32)
        self.series[:, 0] = np.arange(self.n_samples)

        self.setup_canvas()
        self.setup_callbacks()
        self.setup_ui()

        # aux = np.diff(self.mesh._vl.flatten(), prepend=0, append=0).astype(np.int32)
        # print(aux[aux != 0])
            # , self._c, self._vl, self._li
        self.vb1.camera.set_range()

    def setup_callbacks(self):
        self.canvas.connect(self.on_key_press)

    def on_key_press(self, e):
        key = e.key.name

        if key == 'Escape':
            self.close()

    def setup_ui(self):
        self.setWindowTitle('Isolines Viewer')

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas.native)

        self.setLayout(vlayout)
        self.show()

    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor=BG_COLOR, parent=self)
        # self.canvas.measure_fps()
        grid = self.canvas.central_widget.add_grid()

        self.vb1 = grid.add_view(
            row=0,
            col=0,
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
            camera = 'arcball'

        )
        self.isolines = self.setup_isolines()
        self.mesh = self.setup_mesh()
        self.vb1.add(self.isolines)
        self.vb1.add(self.mesh)

    def setup_mesh(self):
        mesh = scene.visuals.Mesh(
            # shading='flat',
            vertices = self.vertices,
            faces = self.faces,
            # meshdata = self.meshdata,
            face_colors = np.broadcast_to(BG_COLOR, (self.faces.shape[0], 4))
        )
        wireframe_filter = vp.visuals.filters.WireframeFilter(width=.5, color=np.array((0.,0.,0.,1.), dtype=np.float32))
        mesh.attach(wireframe_filter)
        
        return mesh

    def setup_isolines(self):

        # base_cmap = vp.color.get_colormap('gist_rainbow')
        # colors = base_cmap[np.linspace(0., 1., num=self.levels.size)]
        isolines = scene.visuals.Isoline(
            vertices = self.vertices,
            tris = self.faces,
            data = self.values,
            levels = self.levels,
            color_lev = BG_COLOR_CONTRAST, # colors,
            width=6,
        )


        return isolines

class SpatioTemporalViewer(QDialog):
    def __init__(self, model, vertices, faces, values, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

        self.model = model
        self.n_samples = values.shape[1]

        # self.max_value = max(abs(values.min()), abs(values.max())).round(decimals=2)
        # self.min_value = -self.max_value
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
        self.sample_text.text = str(value)
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
                print(idx)
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
        self.canvas = scene.SceneCanvas(show=True, bgcolor=BG_COLOR, parent=self)
        # self.canvas.measure_fps()
        grid = self.canvas.central_widget.add_grid()

        self.vb1 = grid.add_view(
            row=0,
            col=0,
            border_color=BG_COLOR_CONTRAST,
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
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
            camera = 'panzoom'

        )
        self.vb2.height_max = 150
        self.vb2.camera.interactive = False

        self.line = self.setup_line()

        self.sample_text = scene.visuals.Text(
            color = BG_COLOR_CONTRAST,
            pos = (0, self.max_value/2)
        )

        self.vb2.add(self.line)
        self.vb2.add(self.sample_text)

    def setup_markers(self):
        scatter = MarkersPicking(
            self.meshdata.get_vertices(),
            BG_COLOR_CONTRAST,
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
            label_color=BG_COLOR_CONTRAST,
            axis_ratio=0.05,
            clim = clim,
        )

        return colorbar

    def setup_mesh(self, cmap, clim):

        mesh = scene.visuals.Mesh(
            shading='flat',
            meshdata = self.meshdata,
        )
        wireframe_filter = vp.visuals.filters.WireframeFilter(width=.1, color=BG_COLOR_CONTRAST)

        mesh.cmap = cmap
        mesh.clim = clim
        mesh.attach(wireframe_filter)
        
        return mesh
class LambdaViewer(QDialog):
    def __init__(self, model, vertices, faces, values, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

        self.model = model
        self.n_lambdas, _, self.n_samples = values.shape

        self.meshdata = vp.geometry.MeshData(
            vertices=vertices,
            faces=faces,
            )

        self.values = values
        self.idx_samples = 0
        self.idx_lambda = 0
        self.previous_idx = 0
        self.series = np.empty((self.n_samples, 2), dtype=np.float32)
        self.series[:, 0] = np.arange(self.n_samples)

        self.setup_canvas()
        self.setup_ui()
        self.setup_callbacks()

        self.set_lambda(0)
        self.set_marker(0)

        self.vb1.camera.set_range()

    def setup_callbacks(self):
        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_mouse_press)
        self.hslider.valueChanged.connect(self.set_lambda)

    def set_time(self, idx):
        self.idx_samples = idx
        self.sample_text.text = str(idx)
        self.meshdata.set_vertex_values(self.values[self.idx_lambda,:,idx])
        self.mesh.mesh_data_changed()

    def set_lambda(self, idx):
        self.idx_lambda = idx
        min_value, max_value = self.values[idx].min(), self.values[idx].max()
        self.cbar.clim = (min_value.round(decimals=2), max_value.round(decimals=2))
        self.mesh.clim = (min_value, max_value)
        self.sample_text.pos = (0, max_value/2)
        self.vb2.camera.set_range(x=(0, self.n_samples), y=self.mesh.clim)
        self.set_time(self.idx_samples)
        self.set_series(self.previous_idx)

    def set_series(self, idx):
        self.series[:,1] = self.values[self.idx_lambda, idx,:]
        self.line.set_data(pos=self.series)

    def on_mouse_release(self, e):
        v = self.canvas.visual_at(e.pos)

        if v == self.vb1:
            idx = self.scatter.pick_marker(self.canvas, e)
            if idx is not None:
                print(idx)
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
        self.setWindowTitle('Lambda Viewer')

        self.hslider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.hslider.setMinimum(0)
        self.hslider.setMaximum(self.n_lambdas-1)
        
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas.native)
        vlayout.addWidget(self.hslider)

        self.setLayout(vlayout)
        self.show()

    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor=BG_COLOR, parent=self)
        # self.canvas.measure_fps()
        grid = self.canvas.central_widget.add_grid()

        self.vb1 = grid.add_view(
            row=0,
            col=0,
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
            camera = 'arcball'

        )
        # scene.visuals.XYZAxis(parent=self.vb1.scene)

        cmap = vp.color.get_colormap('turbo')

        self.cbar = self.setup_colorbar(cmap)
        self.mesh = self.setup_mesh(cmap)
        self.scatter = self.setup_markers()

        self.vb1.add(self.mesh)
        self.vb1.add(self.scatter)

        wcbar = grid.add_widget(
            widget=self.cbar,
            row=0,
            col=1,
        )
        wcbar.width_max = 60

        self.vb2 = grid.add_view(
            row=1,
            col=0,
            col_span=2,
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
            camera = 'panzoom'

        )
        self.vb2.height_max = 150
        self.vb2.camera.interactive = False

        self.line = self.setup_line()

        self.sample_text = scene.visuals.Text(
            color = BG_COLOR_CONTRAST,
        )

        self.vb2.add(self.line)
        self.vb2.add(self.sample_text)

    def setup_markers(self):
        scatter = MarkersPicking(
            self.meshdata.get_vertices(),
            BG_COLOR_CONTRAST,
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

    def setup_colorbar(self, cmap):
        scene.widgets.colorbar.ColorBarVisual.text_padding_factor = 2.5
        colorbar = scene.widgets.ColorBarWidget(
            cmap,
            'left',
            label_color=BG_COLOR_CONTRAST,
            axis_ratio=0.05,
        )

        return colorbar

    def setup_mesh(self, cmap):

        mesh = scene.visuals.Mesh(
            shading='flat',
            meshdata = self.meshdata,
        )
        wireframe_filter = vp.visuals.filters.WireframeFilter(width=.1, color=BG_COLOR_CONTRAST)

        mesh.cmap = cmap
        mesh.attach(wireframe_filter)
        
        return mesh

class AtRtViewer(QDialog):
    def __init__(self, model, vertices, faces, values, times, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

        self.model = model
        self.n_steps = 8
        self.n_samples = values.shape[1]
        self.min_time, self.max_time = times.min(), times.max()
        self.min_value, self.max_value = values.min().round(decimals=2), values.max().round(decimals=2)
        self.times = times.flatten()
        self.center = np.mean(vertices, axis = 0)

        self.meshdata = vp.geometry.MeshData(
            vertices=vertices,
            faces=faces,
            vertex_values=self.times
        )

        self.values = values
        self.previous_idx = 0
        self.series = np.empty((self.n_samples, 2), dtype=np.float32)
        self.series[:, 0] = np.arange(self.n_samples)

        self.setup_canvas()
        self.setup_ui()
        self.setup_callbacks()

        self.set_time(self.times[0])
        self.set_marker(0)

        self.vb1.camera.set_range()
        self.vb2.camera.set_range(y=(self.min_value, self.max_value))

        # self.set_range()

    def setup_callbacks(self):
        self.canvas.connect(self.on_key_press)
        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_mouse_press)

    def on_mouse_move(self, e):
        if e.is_dragging:
            self.azimuth_label.text = f'A : {self.vb1.camera.azimuth:.1f}'
            self.elevation_label.text = f'E : {self.vb1.camera.elevation:.1f}',


    def on_mouse_press(self, e):
        self.line.pick_cursor(self.canvas, e)

    def set_time(self, idx):
        self.sample_text.text = str(self.times[idx])
        self.line.cursor.set_data(self.times[idx])

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
        self.set_time(idx)
        self.previous_idx = idx

    def on_key_press(self, e):
        key = e.key.name

        if key == 'Escape':
            self.close()

        if key == '+':
            self.n_steps += 1
            cmap = self.get_cmap(self.n_steps)
            self.cbar.cmap = cmap
            self.cbar.clim = self.cbar.clim # this update the colorbar
            self.mesh.cmap = cmap

        if key == '-':
            self.n_steps = self.n_steps - 1 if self.n_steps > 2 else 2
            cmap = self.get_cmap(self.n_steps)
            self.cbar.cmap = cmap
            self.cbar.clim = self.cbar.clim # this update the colorbar
            self.mesh.cmap = cmap

        if key == 'N':
            self.scatter.visible = not self.scatter.visible

        if key == 'M':
            self.wireframe_filter.enabled = not self.wireframe_filter.enabled
            self.mesh.update()

        if key == 'B':
            self.wcbar.visible = not self.wcbar.visible

        if key == 'V':
            self.vb2.visible = not self.vb2.visible

        if key == 'C':
            self.tool_bar_grid.visible = not self.tool_bar_grid.visible

        if key == 'S':
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            img = self.screen().grabWindow(self.winId())
            # img.save(f'screenshots/{dt}.ppm', 'ppm', quality=100)
            img.save(f'screenshots/{dt}.bmp', 'bmp', quality=100)
            # img.save(f'screenshots/{dt}.pgm', 'pgm', quality=100)

            # img = vp.gloo.util._screenshot((0, 0, self.canvas.size[0], self.canvas.size[1]))
            # vp.io.imsave(f'vp_{dt}', img, format='eps')

    def setup_ui(self):
        self.setWindowTitle('AT/RT Viewer')

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas.native)

        self.setLayout(vlayout)
        self.show()

    def get_cmap(self, steps):
        # base_cmap = vp.color.get_colormap('Greys')
        base_cmap = vp.color.get_colormap('gist_rainbow')
        colors = base_cmap[np.linspace(0., 1., num=steps)]
        cmap = vp.color.Colormap(colors, interpolation='zero')
        return cmap 

    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor=BG_COLOR, parent=self)
        # self.canvas.measure_fps()
        grid = self.canvas.central_widget.add_grid()

        self.vb1 = grid.add_view(
            row=0,
            col=1,
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
            camera = scene.cameras.TurntableCamera(fov=0, elevation=0, azimuth=90)

        )

        self.azimuth_label = scene.widgets.Label(
            f'A : {self.vb1.camera.azimuth:.1f}',
            anchor_x = 'center',
            color = BG_COLOR_CONTRAST,
        )

        self.elevation_label = scene.widgets.Label(
            f'E : {self.vb1.camera.elevation:.1f}',
            anchor_x = 'center',
            color = BG_COLOR_CONTRAST,
        )

        self.axis = scene.visuals.XYZAxis(
            # color = BG_COLOR_CONTRAST,
            # width = 10
        )


        self.tool_bar_grid = grid.add_grid(
            row=0,
            col=0,
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
        )
        self.tool_bar_grid.width_max = 100

        self.tool_bar_grid.add_widget(
            widget=self.azimuth_label,
            row=0,
            col=0,
        )

        self.tool_bar_grid.add_widget(
            widget=self.elevation_label,
            row=1,
            col=0,
        )


        self.tool_bar_grid.add_widget(
            row=3,
            col=0,
            row_span=9
        )

        clim = (self.min_time, self.max_time)
        cmap = self.get_cmap(self.n_steps)
        self.cbar = self.setup_colorbar(cmap, clim)
        self.mesh = self.setup_mesh(cmap, clim)
        self.scatter = self.setup_markers()

        self.vb1.add(self.mesh)
        self.vb1.add(self.scatter)

        self.vb3 = self.tool_bar_grid.add_view(
            row=2,
            col=0,
            border_color=BG_COLOR_CONTRAST,
            border_width=.0,
            camera = 'turntable',
            row_span=2

        )

        self.vb3.add(self.axis)
        self.vb3.camera.link(self.vb1.camera, props=('azimuth', 'elevation'))

        self.wcbar = grid.add_widget(
            widget=self.cbar,
            row=0,
            col=2,
        )
        self.wcbar.width_max = 60

        self.vb2 = grid.add_view(
            row=1,
            col=0,
            col_span=3,
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
            camera = 'panzoom'

        )
        self.vb2.height_max = 150
        self.vb2.camera.interactive = False

        self.line = self.setup_line()

        self.sample_text = scene.visuals.Text(
            color = BG_COLOR_CONTRAST,
            pos = (0, self.max_value/2)
        )

        self.vb2.add(self.line)
        self.vb2.add(self.sample_text)

    def setup_markers(self):
        scatter = MarkersPicking(
            self.meshdata.get_vertices(),
            BG_COLOR_CONTRAST,
            parent=self.vb1.scene,
            size=7,
            click_radius=5
        )
        return scatter

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
            label_color=BG_COLOR_CONTRAST,
            axis_ratio=0.05,
            clim = clim,
        )

        return colorbar

    def setup_mesh(self, cmap, clim):

        mesh = scene.visuals.Mesh(
            shading='flat',
            meshdata = self.meshdata,
            color=BG_COLOR_CONTRAST
        )
        self.wireframe_filter = vp.visuals.filters.WireframeFilter(width=1., color=BG_COLOR_CONTRAST)

        mesh.cmap = cmap
        mesh.clim = clim
        mesh.attach(self.wireframe_filter)
        
        return mesh

class TemporalViewer(QDialog):

    def __init__(self, model, matrices, colors, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

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
            border_color=BG_COLOR_CONTRAST,
            border_width=0.
        )

        self.text_view.width_max = 60
        self.text_view.camera = 'panzoom'
        self.text_view.camera.interactive = False

        for i in range(self.n_channels):
            scene.visuals.Text(
                text=f'{i}',
                pos=(0, -i*self.offset_series),
                color=BG_COLOR_CONTRAST,
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
            face_color=BG_COLOR_CONTRAST,
            parent=self.text_view.scene
        )

    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor=BG_COLOR, parent=self)
        # self.canvas.measure_fps()
        self.grid = self.canvas.central_widget.add_grid()
        self.setup_panel()
        self.setup_plot()

    def setup_plot(self):

        self.view = self.grid.add_view(
            row=0,
            col=1,
            border_color=BG_COLOR_CONTRAST,
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

        if len(modifiers) > 0: # zoom en y.
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
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

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

class MeshViewer(QDialog):
    def __init__(self, model, vertices, faces, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

        self.model = model
        self.meshdata = vp.geometry.MeshData(
            vertices=vertices,
            faces=faces,
            )

        self.setup_canvas()
        self.setup_ui()
        self.setup_callbacks()

        self.vb1.camera.set_range()

    def setup_callbacks(self):
        self.canvas.connect(self.on_key_press)

    def on_key_press(self, e):
        key = e.key.name

        if key == 'Escape':
            self.close()

    def setup_ui(self):
        self.setWindowTitle('Mesh Viewer')

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas.native)

        self.setLayout(vlayout)
        self.show()

    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor=BG_COLOR, parent=self)
        # self.canvas.measure_fps()
        grid = self.canvas.central_widget.add_grid()

        self.vb1 = grid.add_view(
            row=0,
            col=0,
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
            camera = 'arcball'

        )
        # scene.visuals.XYZAxis(parent=self.vb1.scene)

        self.mesh = self.setup_mesh()
        self.vb1.add(self.mesh)

    def setup_mesh(self):

        mesh = scene.visuals.Mesh(
            shading='flat',
            meshdata = self.meshdata,
        )
        wireframe_filter = vp.visuals.filters.WireframeFilter(width=.1, color=BG_COLOR_CONTRAST)
        mesh.attach(wireframe_filter)
        
        return mesh

class ScatterViewer(QDialog):
    def __init__(self, model, vertices, parent=None):
        super().__init__(f=Qt.WindowMaximizeButtonHint, parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

        self.model = model
        self.scatter = scene.visuals.Markers(
            pos=vertices,
            # pos=np.zeros((1,3), dtype=np.float32),
            size=5,
            face_color=WHITE,
            edge_width=0,
        )

        self.setup_canvas()
        self.setup_ui()
        self.setup_callbacks()

        self.vb1.camera.set_range()

    def setup_callbacks(self):
        self.canvas.connect(self.on_key_press)

    def on_key_press(self, e):
        key = e.key.name

        if key == 'Escape':
            self.close()

    def setup_ui(self):
        self.setWindowTitle('Mesh Viewer')

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.canvas.native)

        self.setLayout(vlayout)
        self.show()

    def setup_canvas(self):
        self.canvas = scene.SceneCanvas(show=True, bgcolor=BG_COLOR, parent=self)
        # self.canvas.measure_fps()
        grid = self.canvas.central_widget.add_grid()

        self.vb1 = grid.add_view(
            row=0,
            col=0,
            border_color=BG_COLOR_CONTRAST,
            border_width=0.,
            camera = 'arcball'

        )
        # scene.visuals.XYZAxis(parent=self.vb1.scene)

        self.vb1.add(self.scatter)

class FPTViewer(QDialog):

    def __init__(self, model, signal, fpt, bad_beats, window, parent=None) -> None:
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose) # free memory on close

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
        self.canvas = scene.SceneCanvas(show=True, bgcolor=BG_COLOR, size=(800,200), parent=self)
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