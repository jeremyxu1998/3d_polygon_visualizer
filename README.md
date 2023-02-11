# 3D Polygon Visualizer
This is a simple program to complete software assessment for Neocis.

## Environment Setup
I choose Python for its fast prototyping capability. Packages used:

```
numpy
bokeh
```

## Quick Start
To run part 1: <br>

To run part 2: <br>

## Part 1
Because the observer/camera is at infinite distance, parallel projection is applied to the canvas. We use rotation matrix to calculate the current 3D location of each vertex, and the x and y value is the coordinate on the canvas.

Based on the tasks, the plotting library must satisfy the following requirements:
- Can plot 2D polygons: vertices, edges and faces, with user defined color
- Has callback API to respond to mouse click and drag event
- Have 2D plotting capability only
The plotting library I choose is [Bokeh](https://bokeh.org/)

Object oriented, separate computation from plotting

## Part 2
