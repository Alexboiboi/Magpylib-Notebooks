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

+++ {"tags": []}

# Extra Model3d

```{code-cell} ipython3
from functools import partial

import ipywidgets as widgets
import magpylib as magpy
import numpy as np
import plotly.graph_objects as go


class myCuboid(magpy.magnet.Cuboid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = np.array([0.0, 0.0, 0.0])
        self.style.model3d.add_trace(
            backend="plotly",
            trace=partial(self.get_scatter_trace, backend="plotly"),
            show=True,
        )
        self.style.model3d.add_trace(
            backend="matplotlib",
            trace=partial(self.get_scatter_trace, backend="matplotlib"),
            show=True,
            coordsargs={"x": "xs", "y": "ys", "z": "zs"},
        )

    def get_scatter_trace(self, backend):
        dim = np.array(self.dimension).astype(float)
        offset = np.array(self.offset).astype(float)
        xyz = np.array(
            [
                [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
                [-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1],
            ]
        )
        xyz = (xyz.T * dim + offset).T
        trace = {
            "type": "scatter3d",
            **{f"{k}": i for i, k in zip(xyz, "xyz")},
            "mode": "lines",
            "line_color": f"rgb{tuple(self.magnetization.astype(int))}",
        }
        if backend == "plotly":
            return trace
        elif backend == "matplotlib":
            trace = {
                "type": "plot",
                **{f"{k}s": i for i, k in zip(xyz, "xyz")},
                "color": tuple(self.magnetization.astype(int) / 255),
            }
            return trace


cuboid = myCuboid(
    magnetization=(1, 0, 0), dimension=(3, 3, 3), position=(10, 0, 0)
).rotate_from_angax(
    np.linspace(0, 180, 36, endpoint=False), "z", anchor=(0, 0, 0), start=0
)

out = widgets.Output()


@out.capture(wait=True, clear_output=True)
def update_fig(
    backend,
    zoom,
    animation,
    showdefault,
    path_show,
    path_frames,
    scale,
    size_x,
    size_y,
    size_z,
    mag_x,
    mag_y,
    mag_z,
    off_x,
    off_y,
    off_z,
):
    cuboid.dimension = size_x, size_y, size_z
    cuboid.magnetization = mag_x, mag_y, mag_z
    cuboid.offset = off_x, off_y, off_z
    cuboid.style.model3d.showdefault = showdefault
    cuboid.style.path.show = path_show
    cuboid.style.path.frames = path_frames
    for t in cuboid.style.model3d.data:
        t.scale = scale
    disp_kwargs = dict(animation=animation, zoom=zoom)
    if "plotly" in backend:
        fig = go.Figure()
        magpy.show(cuboid, canvas=fig, **disp_kwargs, backend="plotly")
        fig.update_layout(height=500, margin_t=0)
        fig.show()
    if "matplotlib" in backend:
        magpy.show(cuboid, **disp_kwargs, backend="matplotlib")


wd = dict(
    backend=widgets.SelectMultiple(
        value=["plotly"], options=["plotly", "matplotlib"], rows=2
    ),
    size_x=widgets.FloatSlider(min=0, max=10, value=5),
    size_y=widgets.FloatSlider(min=0, max=10, value=5),
    size_z=widgets.FloatSlider(min=0, max=10, value=5),
    scale=widgets.FloatLogSlider(min=-1, max=1, base=10, step=0.1),
    mag_x=widgets.IntSlider(min=0, max=255, value=0),
    mag_y=widgets.IntSlider(min=0, max=255, value=0),
    mag_z=widgets.IntSlider(min=0, max=255, value=0),
    off_x=widgets.IntSlider(min=-10, max=10, value=0),
    off_y=widgets.IntSlider(min=-10, max=10, value=0),
    off_z=widgets.IntSlider(min=-10, max=10, value=0),
    showdefault=widgets.Checkbox(value=False),
    animation=widgets.Checkbox(value=False),
    zoom=widgets.FloatSlider(min=0, max=10, value=0.5),
    path_show = widgets.Checkbox(value=True),
    path_frames=widgets.IntSlider(min=0, max=10),
)


def on_cont_up_change(change=None):
    for w in wd.values():
        w.continuous_update = change.new


continuous_update_checkbox = widgets.Checkbox(
    description="continuous slider update", value=True
)
continuous_update_checkbox.observe(on_cont_up_change, names="value")
continuous_update_checkbox.value = False
w = widgets.interactive(update_fig, **wd)
w.update()
out.layout.min_width='600px'
display(widgets.HBox([w, out], layout=dict(flex_flow='wrap')), continuous_update_checkbox)
```

```{code-cell} ipython3

```
