# 3D Polygon Visualizer
This program is to complete software assessment for Neocis.

## Environment Setup
I choose Python for its fast prototyping capability. Packages used:
```
numpy
bokeh
```

## Quick Start
To run part 1: <br>
`bokeh serve --show main.py`

To run part 2: <br>

## Part 1
Because the observer/camera is at infinite distance, parallel projection is applied to the canvas. We use rotation matrix to calculate the current 3D location of each vertex, and the x and y values are the coordinate on the canvas.

Based on the tasks, the plotting library must satisfy the following requirements:
- Can plot 2D polygons: vertices, edges and faces, with user defined color
- Has callback API to respond to mouse click and drag action
- Have 2D plotting capability only

The plotting library I choose is [Bokeh](https://bokeh.org/), when the program executes, it will open a browser window to show the plot.

The most tricky part of this assessment is to define a custom callback function to the mouse drag action, which Bokeh supports through its [Pan, PanEnd and PanStart events](https://docs.bokeh.org/en/latest/docs/reference/events.html#bokeh.events.Pan). Bokeh supports two types of callback function:
- [JavaScript](https://docs.bokeh.org/en/latest/docs/user_guide/interaction/js_callbacks.html): Still calling the Python code to execute, but the callback function needs to be written in JavaScript, which has limited functionality
- [Python](https://docs.bokeh.org/en/latest/docs/user_guide/interaction/python_callbacks.html): Need to create a Bokeh server to run the program (thus the command line will start with `bokeh serve --show`), but the advantage is that callback function can be written in Python, and can utilize numpy to accelerate matrix operations

Therefore, I choose the Python callback method, with its ["Single module format"](https://docs.bokeh.org/en/latest/docs/user_guide/server/app.html#single-module-format)

The callback function is defined in `Window` class, and is called when the mouse is dragged. The callback function will transfer `delta_x` and `delta_y` to the rotation angle, and use rotation matrices to calculate the new 3D location of each vertex, then update the [ColumnDataSource](https://docs.bokeh.org/en/latest/docs/user_guide/data.html#providing-data-as-a-columndatasource) for elements in the plot. The re-plotting will be triggered automatically. To ensure the same rotation scale among each drag, an accumulated rotation matrix is stored, and updated at each `PanEnd` event.

## Part 2
