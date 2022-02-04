---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Compound Class example

```{code-cell} ipython3
from functools import partial

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
from magpylib.display.plotly import make_BaseCylinderSegment


class MagneticWheel(magpy.Collection):
    """creates a basic Collection Compound object with a rotary arrangement of cuboid magnets"""

    def __init__(self, *children, cubes=6, height=10, diameter=36, **style_kwargs):
        super().__init__(*children, **style_kwargs)
        self.update(cubes=cubes, height=height, diameter=diameter)
        self.style.model3d.add_trace(
            backend="plotly",
            trace=partial(self.get_trace3d, backend="plotly"),
            show=True,
        )
        self.style.model3d.add_trace(
            backend="matplotlib",
            trace=partial(self.get_trace3d, backend="matplotlib"),
            show=True,
            coordsargs={"x": "args[0]", "y": "args[1]", "z": "args[2]"},
        )

    def update(self, cubes=None, height=None, diameter=None):
        """updates the magnetic weel object"""
        self.reset_path()
        self.cubes = cubes if cubes is not None else self.cubes
        self.height = height if height is not None else self.height
        self.diameter = diameter if diameter is not None else self.diameter
        create_cube = lambda: magpy.magnet.Cuboid(
            magnetization=(1, 0, 0),
            dimension=[self.height] * 3,
            position=(self.diameter / 2, 0, 0),
        )
        ref_cube = create_cube().rotate_from_angax(
            np.linspace(0.0, 360.0, self.cubes, endpoint=False),
            "z",
            anchor=(0, 0, 0),
            start=0,
        )
        children = []
        for ind in range(cubes):
            s = create_cube()
            s.position = ref_cube.position[ind]
            s.orientation = ref_cube.orientation[ind]
            children.append(s)
        self.children = children
        return self

    def get_trace3d(self, backend):
        trace_plotly = make_BaseCylinderSegment(
            r1=max(self.diameter/2-self.height, 0),
            r2=self.diameter/2 + self.height,
            h=self.height,
            phi1=0,
            phi2=360,
        )
        opacity = 0.5
        color = "blue"
        trace_plotly = {**trace_plotly, **{"opacity": opacity}}
        if backend == "plotly":
            return trace_plotly
        elif backend == "matplotlib":
            x, y, z, i, j, k = [trace_plotly[k] for k in "xyzijk"]
            triangles = np.array([i, j, k]).T
            trace_mpl = dict(type="plot_trisurf", args=(x, y, z), triangles=triangles)
            return {**trace_mpl, **{"alpha": opacity}}


wheels = []

for ind, cubes in enumerate((3,6,12)):
    diameter = cubes * 5
    wheel = MagneticWheel(
        cubes=cubes, height=10, diameter=diameter, style_name=f"Magnetic Wheel {ind+1}"
    )
    wheel.move((diameter, 0, -diameter * 5))
    wheel.rotate_from_angax(90, 'x')
    wheel.rotate_from_angax(
        np.linspace(0, 360, 20, endpoint=False), "z", start=0, anchor=0
    )
    wheel.set_children_styles(path_show=False)
    wheels.append(wheel)
magpy.show(*wheels, style_path_frames=5, backend="plotly")
```
