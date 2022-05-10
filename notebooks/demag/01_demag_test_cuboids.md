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
from demag_functions import apply_demag, match_pairs
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

# compute field before demag
B0 = sensor.getB(COLL0)
```

```{code-cell} ipython3
coll = COLL0
src_list = coll.sources_all
from scipy.spatial.transform import Rotation as R

num_of_src = len(src_list)
pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
rotQ0 = [src.orientation.as_quat() for src in src_list]
rot0 = R.from_quat(rotQ0)
mag0 = [src.magnetization for src in src_list]
dim0 = [src.dimension for src in src_list]

num_of_pairs = len(src_list) ** 2
pos2 = np.tile(pos0, (len(pos0), 1)) - np.repeat(pos0, len(pos0), axis=0)
rot0Q1 = np.tile(rotQ0, (len(rotQ0), 1))
rot0Q2 = np.repeat(rotQ0, len(rotQ0), axis=0)
# checking relative orientation is very expensive, often not worth the savings
rot2 = (
    (R.from_quat(rot0Q1) * R.from_quat(rot0Q2)).as_matrix().reshape((num_of_pairs, -1))
)
dim2 = np.tile(dim0, (len(dim0), 1)) - np.repeat(dim0, len(dim0), axis=0)
mag2 = np.tile(mag0, (len(mag0), 1)) - np.repeat(mag0, len(mag0), axis=0)
prop = (np.concatenate([pos2, rot2, dim2, mag2], axis=1) + 1e-9).round(8)
uniq, unique_inds, unique_inv_inds = np.unique(
    pos2, return_index=True, return_inverse=True, axis=0
)
```

```{code-cell} ipython3
bary2 = np.concatenate(
    [np.tile(pos0, (len(pos0), 1)), np.repeat(pos0, len(pos0), axis=0)], axis=1
).reshape((-1, 2, 3))


fig = go.FigureWidget()
magpy.show(coll, canvas=fig, style_opacity=0.1)
fig.add_scatter3d()
inds = unique_inds[unique_inv_inds]
uniq, counts = np.unique(inds, return_counts=True)


def update_fig(group_index):
    count = counts[group_index]
    ind = uniq[group_index]
    pos4 = bary2[inds == ind]
    path = np.empty((len(pos4) * 3, 3))
    path[::3] = pos4[:, 0, :]
    path[1::3] = pos4[:, 1, :]
    path[2::3] = None
    trace = dict(
        x=path[:, 0],
        y=path[:, 1],
        z=path[:, 2],
        mode="lines+markers",
        line_color=np.repeat(range(count), 3),
        marker_color=np.repeat(range(count), 3),
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


dropdown = widgets.Dropdown(description="Group index", style=dict(description_width="auto"))
update_dropdown([3, max(counts)])
thresh_widget = widgets.IntRangeSlider(
    min=2, max=max(counts), description="Counts threshold", style=dict(description_width="auto")
)
widgets.interactive(update_fig, group_index=dropdown)
widgets.interactive(update_dropdown, tresh=thresh_widget)

widgets.HBox([widgets.VBox([dropdown, thresh_widget]), fig])
```

```{raw-cell}
a = ["c", "a", "b", "c"]

u, i, v, c = np.unique(
    a, return_index=True, return_inverse=True, return_counts=True, axis=0
)
display(
    f"original: {a}",
    f"unique: {u}",
    f"index: {i}",
    f"inverse: {v}",
    f"counts: {c}",
    f"index[inverse] {i[v]}",
    f"unique[inverse] {u[v]}",
)
```

```{raw-cell}
# apply demag
colls = [COLL0.copy(style_label="No Demag")]
colls.append(
    apply_demag(COLL0, xi_vector, inplace=False, style={"label": "Full demag"})
)
# colls.append(
#     apply_demag(
#         COLL0,
#         xi_vector,
#         inplace=False,
#         split=20,
#         style={"label":"Full demag with splitting (=20)"},
#     )
# )
colls.append(
    apply_demag(
        COLL0,
        xi_vector,
        inplace=False,
        pairs_matching=True,
        style={"label": "Pairs matching"},
    )
)
for max_dist in [2, 3, 5, 10, 20]:
    colls.append(
        apply_demag(
            COLL0,
            xi_vector,
            inplace=False,
            max_dist=max_dist,
            style={"label": f"Max distance matching (={max_dist:02d})"},
        )
    )
```

```{raw-cell}
B_cols = ["Bx [mT]", "By [mT]", "Bz [mT]"]


def read_FEM_data(file, source_type):
    df = pd.read_csv(file, delimiter=",")
    df.columns = ["Distance [mm]"] + B_cols
    df = df[["Distance [mm]"] + B_cols]
    df[B_cols] *= 1000
    df[["Distance [mm]"]] -= 4
    df["Source_type"] = source_type
    return df


def get_magpylib_data(collection, sensor):
    B0 = collection.getB(sensor)
    df = pd.DataFrame(
        data=B0,
        columns=B_cols,
    )
    df["Distance [mm]"] = sensor.position[:, 0]
    df["Source_type"] = collection.style.label
    return df


df = pd.concat(
    [
        read_FEM_data("FEMdata_test_cuboids.csv", "FEM (ANSYS)"),
        *[get_magpylib_data(coll, sensor) for coll in colls],
    ]
).sort_values(["Source_type", "Distance [mm]"])
```

```{raw-cell}
colls[0].show()
```

```{raw-cell}
px_kwargs = dict(
    x="Distance [mm]",
    y=B_cols,
    facet_col="variable",
    color="Source_type",
    line_dash="Source_type",
    # height=600,
    facet_col_spacing=0.05,
)
fig1 = px.line(
    df,
    title="FEM vs Magpylib vs Magpylib+Demag",
    **px_kwargs,
)
fig1.update_yaxes(matches=None, showticklabels=True)

df_diff = df.copy()
ref = "Full demag"
for st in df_diff["Source_type"].unique():
    df_diff.loc[df_diff["Source_type"] == st, B_cols] -= df.loc[
        df["Source_type"] == ref, B_cols
    ].values

fig2 = px.line(
    dff,
    title=f"FEM vs Magpylib vs Magpylib+Demag (diff vs {ref})",
    **px_kwargs,
)
fig2.update_yaxes(matches=None, showticklabels=True)
display(fig1, fig2)
```
