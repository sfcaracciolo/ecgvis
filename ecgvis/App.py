from PySide6.QtGui import QKeyEvent, QScreen
from ecgvis.Forms import *
from ecgvis.Panels import TasksContainer, ZarrExplorer
from ecgvis.Models import ZarrModel
import sys
import zarr
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget
from datetime import datetime

# ZARR_PATH = [   '/media/santiago/datos/Repositorios/pyecgi_struct/src/preprocessing ds_0/ds.zarr',
#                 '/media/santiago/datos/Repositorios/pyecgi_struct/src/preprocessing ds_1/ds.zarr',
#                 '/media/santiago/datos/Repositorios/pyecgi_struct/src/preprocessing ds_4/ds.zarr',
#                 '/media/santiago/datos/Repositorios/pyecgi_struct/src/preprocessing ds_13/ds.zarr',
#                 '/media/santiago/datos/sync/Tesis doctoral/Artículos propios/WR/wr_db_mendieta.zarr',
#                 '/media/santiago/datos/sync/Tesis doctoral/Artículos propios/WR/wr_db_caracciolo.zarr'
#             ]

# DATASET = int(sys.argv[-1])

class ecgvis(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app

        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle('ecgvis')
        self.model = ZarrModel()
        zarrExplorer = ZarrExplorer(parent=self)
        zarrExplorer.treeView.setModel(self.model)

        zarrDock = QDockWidget()
        zarrDock.setWindowTitle('Data Explorer')
        zarrDock.setAllowedAreas(Qt.LeftDockWidgetArea)
        zarrDock.setFeatures(QDockWidget.DockWidgetMovable)
        zarrDock.setWidget(zarrExplorer)
        self.addDockWidget(Qt.LeftDockWidgetArea, zarrDock)

        # Create forms
        forms = [
            # (BadChannelsForm(self.model), 'Bad Channels'),
            # (DictionaryForm(self.model), 'Dictionaries'),
            # (SpatialForm(self), 'Spatial Viewer'),
            (TemporalViewerForm(self.model), 'Temporal Viewer'),
            (MatrixViewerForm(self.model), 'Matrix Viewer'),
            (SpatioTemporalViewerForm(self.model), 'SpatioTemporal Viewer'),
            (AtRtViewerForm(self.model), 'AT/RT Viewer'),
            # (ExperimentForm(self.model), 'Experiment Analizer'),
            # (InverseProblemForm(self.model), 'Inverse Problem'),
            # (NearestNodesForm(self.model), 'Nearest Nodes'),
            # (PostProcessingForm(self.model), 'Pos Processing'),
            (NonlinearNotchForm(self.model), 'Nonlinear Adaptive Notch Filter'),
            (MedianFilterForm(self.model), 'Baseline Drift Removal Median Filter'),
            (SplineFilterForm(self.model), 'Baseline Drift Removal Spline Filter'),
            (IsolineCorrectionForm(self.model), 'Isoline Correction'),
            (LaplaceInterpolationForm(self.model), 'Laplace Spatial Interpolation'),
            # (SparseDictBuilderForm(self.model), 'Sparse Dictionary Builder'),
            (FPTForm(self.model), 'Fiducials Point Table Viewer'),
        ]

        # Menu Bar
        menuBar = self.menuBar()
        self.fileMenu = menuBar.addMenu('&File')
        self.tasksMenu = menuBar.addMenu('&Tasks')

        # Tasks Container
        tasksContainer = TasksContainer(forms, parent=self)

        tasksDock = QDockWidget()
        tasksDock.setWindowTitle('Tasks Container')
        tasksDock.setAllowedAreas(Qt.RightDockWidgetArea)
        tasksDock.setFeatures(QDockWidget.DockWidgetMovable)
        tasksDock.setWidget(tasksContainer)
        self.addDockWidget(Qt.RightDockWidgetArea, tasksDock)

        self.setCentralWidget(QWidget())
        self.show()
        self.resize(500, 700)

        # load default zarr path
        path = '/media/santiago/datos/db.zarr'
        self.model.setZarrRoot(path)
        zarrExplorer.treeView.setRootIndex(self.model.index(path))

    def screenshot(self):
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img = self.screen().grabWindow(self.winId())
        img.save(f'screenshots/{dt}.png', 'png', quality=100)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_S:
            self.screenshot()
        return super().keyPressEvent(event)

def run():
    app = QApplication(sys.argv)
    viewer = ecgvis(app)
    sys.exit(app.exec())