import os
import pathlib
from typing import Any, Union
import typing
from PySide6 import QtGui
from PySide6.QtCore import QAbstractListModel, QAbstractTableModel, QDir, QSortFilterProxyModel, Qt, QPersistentModelIndex, QModelIndex, QMimeData, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileSystemModel
import zarr 
import numpy as np
from ecgvis import resources
from ecgvis.Constants import * 

class MatrixModel(QAbstractTableModel):
    def __init__(self, data: np.ndarray, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = data

    def rowCount(self, parent=None) -> int:
        return self._data.shape[0]

    def columnCount(self, parent=None) -> int:
        return self._data.shape[1]

    def data(self, index: QModelIndex, role: int) -> typing.Any:
        if role == Qt.DisplayRole:
            i, j = index.row(), index.column()
            value = self._data[i,j]
            return float(value)

    def setData(self, index: Union[QModelIndex, QPersistentModelIndex], value: Any, role: int) -> bool:
        if role == Qt.EditRole:
            i, j = index.row(), index.column()
            try:
                self._data[i,j] = value
            except ValueError:
                return False
            else:
                return True

    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> Any:
        if role == Qt.DisplayRole:
            return super().headerData(section, orientation, role=role)-1
        return super().headerData(section, orientation, role=role)

    def flags(self, index: Union[QModelIndex, QPersistentModelIndex]) -> Qt.ItemFlags:
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def insertRows(self, row: int, count: int, parent: Union[QModelIndex, QPersistentModelIndex]) -> bool:
        row = self.rowCount() if row == -1 else row
        self.beginInsertRows(parent, row, row+count-1)
        values = np.zeros(count, dtype=self._data.dtype)
        self._data = np.insert(self._data, row, values, axis=0)
        self.endInsertRows()
        return True

    def insertColumns(self, column: int, count: int, parent: Union[QModelIndex, QPersistentModelIndex]) -> bool:
        column = self.columnCount() if column == -1 else column
        self.beginInsertColumns(parent, column, column+count-1)
        values = np.zeros(count, dtype=self._data.dtype)
        self._data = np.insert(self._data, column, values, axis=1)
        self.endInsertColumns()
        return True

class CursorsModel(QAbstractListModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self._data = []
        
    def rowCount(self, parent=None) -> int:
        return len(self._data)
    
    def data(self, index: QModelIndex, role: int) -> typing.Any:
        row = index.row()
        cursor = self._data[row]

        if role == Qt.DisplayRole:
            value = cursor.transform.translate[0]
            return float(value)
        
    def _remove_cursor(self, row):
        cursor = self._data[row]
        self._data.remove(cursor)
        self.layoutChanged.emit()

    def _add_cursor(self, cursor):
        self._data.append(cursor)
        self.layoutChanged.emit()

    def get_values(self):
        size = self.rowCount()
        cursor_array = np.empty(size, dtype=np.float32)
        for i in range(size):
            index = self.index(i)
            cursor_array[i] = self.data(index, Qt.DisplayRole)
        return cursor_array

    def clear(self):
        self._data = []
        self.layoutChanged.emit()
        
class ZarrModel(QFileSystemModel):

    def __init__(self):
        super().__init__()
        self.setZarrRoot()
        self.setFilter(QDir.Dirs | QDir.NoDotAndDotDot)
        self.icon_file = QIcon(":/icons/file.png")

    def setZarrRoot(self, path=None):
        if path is None:
            self.zarr_root = None
        else:
            self.zarr_root = zarr.open(path, mode='r+')
            self.setRootPath(path)

    def data(self, index: Union[QModelIndex, QPersistentModelIndex], role: int) -> Any:
        if self.zarr_root is not None:
            if role == Qt.DecorationRole:
                if self.is_array(index):
                    return  self.icon_file

            return super().data(index, role=role)
        return False

    def flags(self, index: Union[QModelIndex, QPersistentModelIndex]) -> Qt.ItemFlags:
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled

    def mimeData(self, indexes: typing.Iterable[QModelIndex]) -> QMimeData:
        mimeData = super().mimeData(indexes)
        try:
            index = indexes[0]
        except:
            pass
        else:
            mimeData.setText(self.get_path(index))
        finally:
            return mimeData

    def is_array(self, index):
        path = self.filePath(index)
        return os.path.exists(os.path.join(path, '.zarray'))

    def get_path(self, index):
        return os.path.relpath(self.filePath(index), self.rootPath())

    def get_info(self, index):
        zpath = self.get_path(index)
        zobj = self.zarr_root[zpath]
        info = zobj.info_items()
        attrs = list(zobj.attrs.asdict().items())
        if attrs:
            info += attrs
        return zarr.util.info_html_report(info)

class FPTModel(MatrixModel):
    def __init__(self, data: np.ndarray, mask: np.ndarray, parent=None) -> None:
        super().__init__(data, parent=parent)
        self.mask = mask
        self._header = (
            'Pon',
            'Ppeak',
            'Poff',
            'QRSon',
            'Q',
            'R',
            'S',
            'QRSoff',
            'res.',
            'Ton',
            'Tpeak',
            'Toff',
            'res.',
        )

    def flags(self, index: Union[QModelIndex, QPersistentModelIndex]) -> Qt.ItemFlags:
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        
    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> typing.Any:

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._header[section]

        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return section

        return super().headerData(section, orientation, role=role)

    def data(self, index: QModelIndex, role: int) -> typing.Any:
        i = index.row()
        if role == Qt.BackgroundRole:
            if not self.mask[i]:
                return QtGui.QColor.fromRgbF(*RED.tolist())

        return super().data(index, role)