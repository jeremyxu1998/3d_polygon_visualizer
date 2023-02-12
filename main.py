import numpy as np
import argparse

from bokeh.plotting import figure, curdoc
from bokeh.layouts import column
from bokeh.models import Range1d, ColumnDataSource, Scatter


class Polygon:
    def __init__(self, obj_path):
        """Load polygon/mesh from txt file"""
        f = open(obj_path, "r")
        lines = f.readlines()
        f.close()
        self.num_vertices, self.num_faces = [int(num) for num in lines[0].split(',')]
        assert self.num_vertices + self.num_faces == len(lines) - 1

        # Here assume the index of vertices start from 1 and are consecutive integers
        # If not, can use a dictionary to store the vertices, but it will be slower to query for complicated meshes
        self.vertices = np.zeros((self.num_vertices+1, 3))
        self.faces = np.zeros((self.num_faces, 3), dtype=np.int32)
        self.all_edges = []

        for l in range(1, self.num_vertices + 1):
            line = lines[l].split(',')
            assert len(line) == 4
            x, y, z = [float(line[i]) for i in range(1, 4)]
            self.vertices[int(line[0])] = np.array([x, y, z])
        
        for l in range(self.num_vertices + 1, self.num_vertices + self.num_faces + 1):
            line = lines[l].split(',')
            assert len(line) == 3
            v1, v2, v3 = [int(vid) for vid in line]
            self.faces[l-self.num_vertices-1] = np.array([v1, v2, v3])
            vs = sorted([v1, v2, v3])
            for edge in ([vs[0], vs[1]], [vs[1], vs[2]], [vs[0], vs[2]]):
                if edge not in self.all_edges:
                    self.all_edges.append(edge)
    
    def get_vertices(self):
        return self.vertices
    
    def get_faces(self):
        return self.faces
    
    def get_edges(self):
        return self.all_edges


class Window:
    def __init__(self, polygon, args):
        self.render_mode = args.render_mode
        self.fill_scale = 1.5
        
        self.rot_matrix = np.eye(3)  # current object rotation WRT coordinate at initialization
        self.accu_rot_matrix = np.eye(3)  # accumulated rotation matrix from past mouse drags
        self.dist_to_ang = 0.01  # mouse pan distance on screen (pixel) to angle of rotation (rad)
                                 # at this default setting, dragging across window width will rotate object for ~360 deg
        
        self.graph = figure(title = "3D Polygon Visualizer", width=args.width, height=args.height)
        self.graph.toolbar.active_drag = None  # disable original pan tool
        self.graph.on_event("pan", self.on_pan)  # custom pan callback to rotate the polygon
        self.graph.on_event("panend", self.on_pan_end)

        self.polygon: Polygon = polygon
        self.init_source()

        curdoc().add_root(column(self.graph))  # add the plot to the document, initialize the window
    
    def init_source(self):
        """initialize all Bokeh data source"""
        coord_2d = self.project_to_canvas()
        # set plotting coordinate range
        max_coord = np.max(np.abs(coord_2d))
        self.graph.x_range = Range1d(-max_coord*self.fill_scale, max_coord*self.fill_scale)
        self.graph.y_range = Range1d(-max_coord*self.fill_scale, max_coord*self.fill_scale)

        if self.render_mode == "frame":
            self.vertex_source = ColumnDataSource(data=dict(x=coord_2d[1:, 0], y=coord_2d[1:, 1]))
            self.vertex_glyph = Scatter(x="x", y="y", size=10, fill_color="#0000ff", marker="circle")
            self.graph.add_glyph(self.vertex_source, self.vertex_glyph)

            all_3d_edges = self.polygon.get_edges()
            edge_source_dict = {}
            for i, edge_ids in enumerate(all_3d_edges):
                edge_2d = coord_2d[edge_ids]  # (start & end, x & y)
                edge_source_dict['x'+str(i)] = edge_2d[:, 0]
                edge_source_dict['y'+str(i)] = edge_2d[:, 1]

            self.edge_source = ColumnDataSource(data=edge_source_dict)
            for i in range(len(all_3d_edges)):
                self.graph.line(x='x'+str(i), y='y'+str(i), source=self.edge_source, line_width=5, line_color="#0000ff")
        
        elif self.render_mode == "surface":
            pass

    def project_to_canvas(self):
        """Project 3D vertices to 2D using rotation matrix applied from window"""
        cur_vertices = np.matmul(self.rot_matrix, self.polygon.get_vertices().T).T
        cur_2d_coord = cur_vertices[:, :2]
        return cur_2d_coord
    
    def on_pan(self, event):
        """Rotate the polygon when the user pans the window"""
        # print("delta_x:", event.delta_x)
        # print("delta_y:", event.delta_y)
        angle_x = event.delta_y * self.dist_to_ang  # rotation around x-axis depends on mouse drag distance along y-axis
        angle_y = event.delta_x * self.dist_to_ang
        rot_mat_x = np.array([[1, 0, 0],
                              [0, np.cos(angle_x), -np.sin(angle_x)],
                              [0, np.sin(angle_x), np.cos(angle_x)]])
        rot_mat_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                              [0, 1, 0],
                              [-np.sin(angle_y), 0, np.cos(angle_y)]])
        # within one drag action, the rotation matrix is based on the accumulated rotation matrix
        self.rot_matrix = np.matmul(np.matmul(rot_mat_x, rot_mat_y), self.accu_rot_matrix)

        # update the Bokeh data source
        coord_2d = self.project_to_canvas()
        self.vertex_source.data = dict(x=coord_2d[1:, 0], y=coord_2d[1:, 1])

        all_3d_edges = self.polygon.get_edges()
        new_edge_source_dict = {}
        for i, edge_ids in enumerate(all_3d_edges):
            edge_2d = coord_2d[edge_ids]
            new_edge_source_dict['x'+str(i)] = edge_2d[:, 0]
            new_edge_source_dict['y'+str(i)] = edge_2d[:, 1]
        self.edge_source.data = new_edge_source_dict
    
    def on_pan_end(self, event):
        """After user finishes one mouse drag,
        update the accumulated rotation to current rotation,
        then reset the self.rot_matrix to identity"""
        self.accu_rot_matrix = self.rot_matrix
        self.rot_matrix = np.eye(3)


# Main function below, but cannot use `if __name__ == "__main__":` because Bokeh run won't trigger it
parser = argparse.ArgumentParser()
parser.add_argument("--obj_path", type=str, default="data/tetrahedron.txt")
parser.add_argument("--render_mode", type=str, default="frame", choices=["frame", "surface"])
parser.add_argument("--width", type=int, default=600)
parser.add_argument("--height", type=int, default=600)
args = parser.parse_args()

polygon = Polygon(args.obj_path)
window = Window(polygon, args)
