
from ecgvis.Constants import *
from vispy import scene
import numpy as np
from vispy.visuals.transforms.linear import STTransform
import itertools
class RegionsLine(scene.Line):
    def __init__(self, size, n_regions=3, n_markers=4, parent=None) -> None:

        self._dataline = np.zeros((size, 2), dtype=np.float32)
        self._dataline[:,0] = np.arange(size)
        self._datapoints = np.zeros((n_regions, 2), dtype=np.float32)
        self._datamarkers = np.zeros((n_markers, 2), dtype=np.float32)
    
        scene.visuals.InfiniteLine(
            pos=0,
            vertical=False,
            color=AMBER,
            parent=parent
        )


        self.markers = scene.visuals.Markers(
            parent=parent
        )

        self.regions = []
        colors = itertools.cycle([LIGHT_RED, LIGHT_BLUE, LIGHT_GREEN])
        for i in range(n_regions):
            r = scene.visuals.LinearRegion(
                pos=self._datapoints[i, :],
                color=next(colors),
                vertical=True,
                parent=parent
            )
            self.regions.append(r)


        super().__init__(
            pos=self._dataline,
            color='white',
            parent=parent
        )
    
    def set_data(self, line, points, markers):
        self._dataline[:,1] = line
        # self.update() # works in linux
        super().set_data(pos=self._dataline) # works in windows.

        self._datapoints[:] = points
        for i, r in enumerate(self.regions):
            r.set_data(pos=self._datapoints[i, :])

        self._datamarkers[:] = markers
        self.markers.set_data(
            pos = self._datamarkers
        )
class LinePicking(scene.Line):
    def __init__(self, parent=None) -> None:

        super().__init__(color=BG_COLOR_CONTRAST, parent=parent)
        
        cursor = scene.visuals.InfiniteLine(
            pos=0.,
            color=AMBER,
            vertical=True,
            parent=self
        )

        cursor.interactive = True
        cursor.transform = STTransform(translate=0)      

        self.unfreeze()
        self.cursor = cursor
        self.active = False
        self.freeze()

    def move_cursor(self, canvas, event):
        pos = None
        if event.is_dragging and self.active:
            tr = canvas.scene.node_transform(self.cursor)
            dx = tr.map(event.pos)[0]
            self.cursor.transform.move(dx)
            pos = np.rint(self.cursor.transform.translate[0])
            pos = int(np.clip(pos, 0, self.pos.shape[0]-1))
        return pos 

    def pick_cursor(self, canvas, event):
        cs = canvas.visuals_at(event.pos, radius=20)
        if len(cs):
            if cs[0] == self.cursor:
                self.active = True
                self.cursor.set_data(color=RED)
                self.cursor.update()

    def drop_cursor(self):
        if self.active:
            self.cursor.set_data(color=AMBER)
            self.cursor.update()
            self.active = False
class Lines(scene.Line):
    def __init__(self, n_lines, line_size, dx, offset, color, parent) -> None:
        self._line_size = line_size
        n_points = line_size*n_lines

        x = np.arange(max(line_size, n_lines), dtype=np.float32)

        connect = np.ones(n_points, dtype=np.bool8)
        connect[line_size-1::line_size] = False

        self._data = np.zeros((n_points, 2), dtype=np.float32)
        self._data[:, 0] = dx*np.tile(x[:line_size], n_lines)

        self._offset = offset*np.repeat(x[:n_lines], line_size)

        super().__init__(
            pos=self._data,
            color=color,
            connect=connect,
            parent=parent
        )

    def set_data(self, index, data):
        y = self._data[:, 1]
        start = index*self._line_size
        stop = start + self._line_size
        y[start:stop] = data
        y[start:stop] -= self._offset[start:stop]

    def set_scale(self, factor):
        y = self._data[:, 1]
        y += self._offset
        y *= factor
        y -= self._offset
        # self.update()
        super().set_data(pos=self._data) # works in windows.

class Isolines(Lines):
    def __init__(self, n_lines, line_size, dx, offset, color, parent) -> None:
        super().__init__(n_lines, line_size, dx, offset, color, parent)
        
        self.transform = STTransform(translate=(0, 0, 1))
        self.order = 1
        
        for i in range(n_lines):
            self.set_data(i)
        
    def set_data(self, index):
        y = self._data[:, 1]
        start = index*self._line_size
        stop = start + self._line_size
        # y[start:stop] = data
        y[start:stop] -= self._offset[start:stop]
class MarkersPicking(scene.Markers):
    """
    Create a 3D scatter plot window that is zoomable and rotateable, with
    markers of a given `symbol` and `size` at the given 3D `positions` and in
    the given RGBA `colors`, formatted as numpy arrays of size Nx3 and Nx4,
    respectively. Takes an optional callback function that will be called with
    the index of a clicked marker and a reference to the Markers visual
    whenever the user clicks a marker (or at most `click_radius` pixels next to
    a marker).
    """
    def __init__(self, positions, base_color, symbol='o', size=4.5, click_radius=2, edge_width=.0, scaling=False, parent=None) -> None:
        super().__init__(parent=parent)

        ids = np.arange(1, len(positions) + 1, dtype=np.uint32).view(np.uint8)
        ids = ids.reshape(-1, 4)

        self.unfreeze()
        self.positions = positions
        self.base_color = base_color
        self.click_radius = click_radius
        self.n_points = positions.shape[0]
        self.colors = np.repeat(base_color[np.newaxis,:], self.n_points, axis=0)
        self.kwargs = dict(symbol=symbol, size=size, edge_color=BG_COLOR_CONTRAST, edge_width=edge_width, scaling=scaling)
        self.ids = np.divide(ids, 255, dtype=np.float32)
        self.freeze()
        # based on https://github.com/vispy/vispy/issues/1189#issuecomment-198597473
        self.set_gl_state('translucent', blend=True, depth_test=True)
        #axis = scene.visuals.XYZAxis(parent=view.scene)
        # set positions and colors
        self.set_data(self.positions, face_color=self.colors, **self.kwargs)
        # prepare list of unique colors needed for picking
    # connect events

    def pick_marker(self, canvas, event):
        if event.button == 1 and self.distance_traveled(event.trail()) <= 2:
            # vispy has some picking functionality that would tell us
            # whether any of the scatter points was clicked, but we want
            # to know which point was clicked. We do an extra render pass
            # of the region around the mouseclick, with each point
            # rendered in a unique color and without blending and
            # antialiasing.
            pos = canvas.transforms.canvas_transform.map(event.pos)
            try:
                self.update_gl_state(blend=False)
                self.antialias = 0
                self.set_data(self.positions, face_color=self.ids, **self.kwargs)
                img = canvas.render((pos[0] - self.click_radius,
                                    pos[1] - self.click_radius,
                                    self.click_radius * 2 + 1,
                                    self.click_radius * 2 + 1),
                                    bgcolor=(0, 0, 0, 0))
            finally:
                self.update_gl_state(blend=True)
                self.antialias = 1
                self.set_data(self.positions, face_color=self.colors, **self.kwargs)
            # We pick the pixel directly under the click, unless it is
            # zero, in which case we look for the most common nonzero
            # pixel value in a square region centered on the click.
            idxs = img.ravel().view(np.uint32)
            idx = idxs[len(idxs) // 2]
            if idx == 0:
                idxs, counts = np.unique(idxs, return_counts=True)
                idxs = idxs[np.argsort(counts)]
                idx = idxs[-1] or (len(idxs) > 1 and idxs[-2])
            # call the callback function
            if idx > 0 and idx <= self.n_points:
                return idx-1
        
        return None
            # if idx > 0:
            #     # subtract one; color 0 was reserved for the background
            #     self.set_color(idx - 1)

    def set_color(self, idx, color=None):
        self.colors[idx] = color if color is not None else self.base_color
        self.set_data(self.positions, face_color=self.colors, **self.kwargs)

    def distance_traveled(self, positions):
        """
        Return the total amount of pixels traveled in a sequence of pixel
        `positions`, using Manhattan distances for simplicity.
        """
        d = np.sum(np.abs(np.diff(positions, axis=0))) if positions is not None else np.inf
        return d