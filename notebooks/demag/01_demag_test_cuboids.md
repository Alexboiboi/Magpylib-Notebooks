---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
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
```

# Setup

```{code-cell} ipython3
# number of target mesh elements
target_cells = 200

# some low quality magnets with different parameters split up into cells
cube1 = magpy.magnet.Cuboid(magnetization=(0, 0, 1000), dimension=(1, 1, 1))
coll1 = mesh_Cuboid(cube1, target_cells)
coll1.move((-1.5, 0, 0))
coll1.xi = 0.3  # mur=1.3

cube2 = magpy.magnet.Cuboid(magnetization=(900, 0, 0), dimension=(1, 1, 1))
coll2 = mesh_Cuboid(cube2, target_cells)
coll2.rotate_from_angax(-45, "y").move((0, 0, 0.2))
coll2.xi = 1.0  # mur=2.0

mx, my = 600 * np.sin(30 / 180 * np.pi), 600 * np.cos(30 / 180 * np.pi)
cube3 = magpy.magnet.Cuboid(magnetization=(mx, my, 0), dimension=(1, 1, 2))
coll3 = mesh_Cuboid(cube3, target_cells)
coll3.move((1.6, 0, 0.5)).rotate_from_angax(30, "z")
coll3.xi = 0.5  # mu3=1.5

# collection of all cells
coll = magpy.Collection(coll1, coll2, coll3)

# sensor
sensor = magpy.Sensor(position=np.linspace((-4, 0, -1), (4, 0, -1), 301))

# compute field before demag
B0 = sensor.getB(COLL0)

coll.show()
```

# Demagnetization computation

```{code-cell} ipython3
# apply demag
colls = [apply_demag(coll, style={"label": "Full demag"})]
```

```{code-cell} ipython3
colls[0].show()
```

# Compare demagnetization methods with FEM

```{code-cell} ipython3
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

```{code-cell} ipython3
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
ref = "FEM (ANSYS)"
for st in df_diff["Source_type"].unique():
    df_diff.loc[df_diff["Source_type"] == st, B_cols] -= df.loc[
        df["Source_type"] == ref, B_cols
    ].values

fig2 = px.line(
    df_diff,
    title=f"FEM vs Magpylib vs Magpylib+Demag (diff vs {ref})",
    **px_kwargs,
)
fig2.update_yaxes(matches=None, showticklabels=True)
display(fig1, fig2)
```
