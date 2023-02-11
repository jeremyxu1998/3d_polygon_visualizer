import numpy as np
import argparse
from bokeh.plotting import figure, show
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
        
        self.polygon: Polygon = polygon
        self.rot_matrix = np.eye(3)
        self.graph = figure(title = "3D Polygon Visualizer", width=args.width, height=args.height)
    
    def update(self):
        coord_2d = self.project_to_canvas()
        # set coordinate range
        max_coord = np.max(np.abs(coord_2d))
        fill_scale = 1.25
        self.graph.x_range = Range1d(-max_coord*fill_scale, max_coord*fill_scale)
        self.graph.y_range = Range1d(-max_coord*fill_scale, max_coord*fill_scale)

        if self.render_mode == "frame":
            self.plot_wireframe(coord_2d)
        elif self.render_mode == "surface":
            pass
        show(self.graph)

    def project_to_canvas(self):
        """Project 3D vertices to 2D using rotation matrix applied from window"""
        cur_vertices = np.matmul(self.rot_matrix, self.polygon.get_vertices().T).T
        cur_2d_coord = cur_vertices[:, :2]
        return cur_2d_coord
    
    def plot_wireframe(self, coord_2d):
        vertex_source = ColumnDataSource(dict(x=coord_2d[1:, 0], y=coord_2d[1:, 1]))
        vertex_glyph = Scatter(x="x", y="y", size=10, fill_color="#0000ff", marker="circle")
        self.graph.add_glyph(vertex_source, vertex_glyph)
        
        for edge_ids in self.polygon.get_edges():
            edge = coord_2d[edge_ids]
            self.graph.line(edge[:, 0], edge[:, 1], line_width=5, line_color="#0000ff")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, default="data/tetrahedron.txt")
    parser.add_argument("--render_mode", type=str, default="frame", choices=["frame", "surface"])
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    polygon = Polygon(args.obj_path)
    window = Window(polygon, args)
    window.update()
