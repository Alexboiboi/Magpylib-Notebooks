---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2
```

```{code-cell} ipython3
import ipywidgets as widgets
import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from demag_functions import apply_demag, filter_distance, match_pairs
from meshing_functions import mesh_Cuboid

magpy.defaults.display.backend = "plotly"

# number of target mesh elements
target_cells = 20

# some low quality magnets with different parameters split up into cells
cube1 = magpy.magnet.Cuboid(magnetization=(0, 0, 1000), dimension=(1, 1, 1))
coll1 = mesh_Cuboid(cube1, target_cells)
coll1.move((-1.5, 0, 0))
xi1 = [0.3] * len(coll1)  # mur=1.3

cube2 = magpy.magnet.Cuboid(magnetization=(900, 0, 0), dimension=(1, 1, 1))
coll2 = mesh_Cuboid(cube2, target_cells)
coll2.rotate_from_angax(-45, "y").move((0, 0, 0.2))
xi2 = [1.0] * len(coll2)  # mur=2.0

mx, my = 600 * np.sin(30 / 180 * np.pi), 600 * np.cos(30 / 180 * np.pi)
cube3 = magpy.magnet.Cuboid(magnetization=(mx, my, 0), dimension=(1, 1, 2))
coll3 = mesh_Cuboid(cube3, target_cells)
coll3.move((1.6, 0, 0.5)).rotate_from_angax(30, "z")
xi3 = [0.5] * len(coll3)  # mu3=1.5

# collection of all cells
COLL0 = magpy.Collection(coll1, coll2, coll3)
xi_vector = np.array(xi1 + xi2 + xi3)

# sensor
sensor = magpy.Sensor(position=np.linspace((-4, 0, -1), (4, 0, -1), 301))
```

+++ {"tags": []}

# Matching Interactions

```{code-cell} ipython3
def show_matching_interactions(collection):
    src_list = collection.sources_all
    params, unique_inds, unique_inv_inds, pos0, rot0 = match_pairs(src_list)

    bary2 = np.concatenate(
        [np.tile(pos0, (len(pos0), 1)), np.repeat(pos0, len(pos0), axis=0)], axis=1
    ).reshape((-1, 2, 3))

    inds = unique_inds[unique_inv_inds]
    uniq, counts = np.unique(inds, return_counts=True)

    fig = go.FigureWidget(layout_height=600, layout_margin_t=0)
    magpy.show(collection, canvas=fig, style_opacity=0.1)
    fig.add_scatter3d()

    def update_fig(group_index):
        count = counts[group_index]
        ind = uniq[group_index]
        pos = bary2[inds == ind]
        path = np.empty((len(pos) * 3, 3))
        path[::3] = pos[:, 0, :]
        path[1::3] = pos[:, 1, :]
        path[2::3] = None
        trace = dict(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines+markers",
            line_color=np.repeat(range(count), 3),
            marker_color=np.repeat(range(count), 3),
            marker_symbol=np.tile(["cross", "circle", "circle-open"], len(counts)),
            name=f"Group_{group_index:03d} ({count} matching interactions)",
        )
        fig.data[1].update(**trace)

    def update_dropdown(tresh):
        c = counts
        opts = [
            (f"Group_{i:03d} ({c[i]} matching interactions)", i)
            for i, _ in enumerate(c)
            if c[i] >= tresh[0] and c[i] <= tresh[1]
        ]
        dropdown.options = opts

    dropdown = widgets.Dropdown(
        description="Group", style=dict(description_width="auto")
    )
    thresh_widget = widgets.IntRangeSlider(
        min=2,
        max=max(counts),
        description="Counts threshold",
        style=dict(description_width="auto"),
        continuous_update=False,
    )

    update_dropdown([3, max(counts)])
    update_fig(0)

    widgets.interactive(update_fig, group_index=dropdown)
    widgets.interactive(update_dropdown, tresh=thresh_widget)

    display(widgets.VBox([dropdown, thresh_widget]), fig)


show_matching_interactions(COLL0)
```

# Filter distance

```{code-cell} ipython3
def show_filter_distance(collection):
    src_list = collection.sources_all
    src_inds = np.array(np.meshgrid(range(len(src_list)), range(len(src_list)))).T.reshape(-1,2)[:,::-1]

    fig = go.FigureWidget(layout_height=600, layout_margin_t=0)
    magpy.show(collection, canvas=fig, style_opacity=0.1)
    fig.add_scatter3d()

    def update_fig(group_index, max_dist):
        params, mask, pos0, rot0 = filter_distance(src_list, max_dist=max_dist)
        bary2 = np.concatenate(
            [np.tile(pos0, (len(pos0), 1)), np.repeat(pos0, len(pos0), axis=0)], axis=1
        ).reshape((-1, 2, 3))

        mask2 = src_inds[:,0]==group_index
        pos = bary2[mask&mask2]
        path = np.empty((len(pos) * 3, 3))
        path[::3] = pos[:, 0, :]
        path[1::3] = pos[:, 1, :]
        path[2::3] = None
        trace = dict(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines+markers",
            line_color=np.repeat(range(len(pos)), 3),
            marker_color=np.repeat(range(len(pos)), 3),
            marker_symbol=np.tile(["cross", "circle", "circle-open"], len(pos)),
            name=f"Source_{group_index:03d} (max_dist interactions)",
        )
        fig.data[1].update(**trace)

    slider_index = widgets.IntSlider(
        min=0,
        max=len(src_list)-1,
        value=0,
        description="Source index",
        style=dict(description_width="auto"),
        continuous_update=False,
    )
    max_dist_slider = widgets.FloatSlider(
        min=0,
        max=10,
        value=1,
        description="Max distance",
        style=dict(description_width="auto"),
        continuous_update=False,
    )


    w = widgets.interactive(update_fig, group_index=slider_index, max_dist=max_dist_slider)
    w.update()

    display(widgets.VBox([slider_index,max_dist_slider]), fig)

show_filter_distance(COLL0)
```
