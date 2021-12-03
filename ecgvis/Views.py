from typing import Optional, Sequence, Union
from PySide6 import QtCore
from PySide6.QtGui import QKeyEvent, QDropEvent
from PySide6.QtCore import QAbstractItemModel, QModelIndex, QPersistentModelIndex, Qt, Signal, QItemSelection
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QTreeView, QWidget
import shutil
from pathlib import Path
        
class MatrixTableView(QTableView):
    def __init__(self, model, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent=parent)
        # self.setCornerButtonEnabled(False)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setModel(model)
        self.setup_callbacks()

    def setup_callbacks(self):
            self.verticalHeader().sectionDoubleClicked.connect(self.model().insertRow)
            self.horizontalHeader().sectionDoubleClicked.connect(self.model().insertColumn)

    def resizeCellsToContents(self):
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Space:
            self.resizeCellsToContents()
        return super().keyPressEvent(event)

class FPTView(QTableView):
    row = Signal(int)

    def __init__(self, model, parent=None) -> None:
        super().__init__(parent=parent)
        self._model = model
        self.setCornerButtonEnabled(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setModel(model)
        self.resizeColumnsToContents()

        self.hheader = self.horizontalHeader()
        self.vheader = self.verticalHeader()

    def selectionChanged(self, selected: QtCore.QItemSelection, deselected: QtCore.QItemSelection) -> None:
        try:
            index = selected.indexes()[0].row()
        except IndexError:
            pass
        else:
            self.row.emit(index)
        return super().selectionChanged(selected, deselected)
        
class ZarrTreeView(QTreeView):

    info_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)

    def setModel(self, model: QAbstractItemModel) -> None:
        super().setModel(model)
        self.setColumnHidden(1, True)
        self.setColumnHidden(2, True)
        self.setColumnHidden(3, True)
        return None

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        try:
            index = self.selectedIndexes()[0]
        except IndexError:
            pass
        else:
            info = self.model().get_info(index)
            self.info_signal.emit(info)

        return super().selectionChanged(selected, deselected)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Delete: # Del 
            model = self.model()
            try:
                index = self.selectedIndexes()[0]
            except IndexError:
                pass
            else:
                path = model.filePath(index)
                shutil.rmtree(path)
        
        return super().keyPressEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        data = event.mimeData()
        if data.hasUrls():
            if len(data.urls()) == 1:
                path = data.urls()[0].path()
                if Path(path).suffix == '.zarr':
                    self.model().setZarrRoot(path)
                    self.setRootIndex(self.model().index(path))
        return super().dropEvent(event)    
