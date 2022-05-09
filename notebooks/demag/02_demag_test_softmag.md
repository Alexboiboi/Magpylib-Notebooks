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
import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.express as px
from demag_functions import apply_demag
from meshing_functions import mesh_Cuboid

elems = 200  # mesh factor

# hard magnet
mag1 = (0, 0, 1000)
dim1 = (1, 1, 2)
cube1 = magpy.magnet.Cuboid(mag1, dim1, (0, 0, 0.5))
col1 = mesh_Cuboid(cube1, elems)
xi1 = [0.5] * len(col1)

# soft magnet
mag2 = (0, 0, 0)
dim2 = (1, 1, 1)
cube2 = magpy.magnet.Cuboid(mag2, dim2, (0, 0, 0))
cube2.rotate_from_angax(angle=45, axis="y", anchor=None).move((1.5, 0, 0))
col2 = mesh_Cuboid(cube2, elems)
xi2 = [3999] * len(col2)

# super collection
COL0 = cube1 + cube2
xi_vector = np.array(xi1 + xi2)
#magpy.show(cube1, cube2, sensors)

# add sensors
sensors = [
    magpy.Sensor(
        position=np.linspace((-4, 0, z), (6, 0, z), 1001),
    )
    for z in [-1, -3, -5]
]

# apply demag
COL1 = col1 + col2

apply_demag(COL1, xi_vector, demag_store=False, demag_load=False)


print("\nAfter demagnetization:")
magpy.show(*COL1, sensors)
```

```{code-cell} ipython3
def read_FEM_data(file, source_type):
    df0 = pd.read_csv(file, delimiter=",")
    df_list = []
    for i, ind in enumerate((1, 3, 5)):
        df = df0.copy()
        B_cols = df.columns[ind : ind + 2].tolist()
        df = df[["Distance [mm]"] + B_cols]
        df[B_cols] *= 1000
        df[["Distance [mm]"]] -= 4
        df.columns = ["Distance [mm]", "Bx [mT]", "Bz [mT]"]
        df["Sensor_num"] = i
        df_list.append(df)
    df = pd.concat(df_list)
    df["Source_type"] = source_type
    return df


def get_magpylib_data(collection, source_type):
    B0 = collection.getB(*sensors)
    df = pd.DataFrame(
        data=B0[..., [0, 2]].swapaxes(0, 1).reshape(-1, 2),
        columns=["Bx [mT]", "Bz [mT]"],
    )
    df["Sensor_num"] = np.repeat(range(B0.shape[1]), B0.shape[0])
    df["Distance [mm]"] = np.array([s.position[:, 0] for s in sensors]).flatten()
    df["Source_type"] = source_type
    return df


df = pd.concat(
    [
        read_FEM_data("FEMdata_test_softmag_coarse.csv", "FEM-coarse"),
        read_FEM_data("FEMdata_test_softmag_fine.csv", "FEM-fine"),
        get_magpylib_data(COL0, "Magpylib-No-Demag"),
        get_magpylib_data(COL1, "Magpylib-With-Demag"),
    ]
)
```

```{code-cell} ipython3
fig = px.line(
    df,
    x="Distance [mm]",
    y=["Bx [mT]", "Bz [mT]"],
    facet_col="Sensor_num",
    facet_row="variable",
    color="Source_type",
    height=600,
    title='FEM vs Magpylib vs Magpylib+Demag',
)
fig.update_yaxes(matches=None, showticklabels=True)
```

```{code-cell} ipython3
dff = df.sort_values(["Source_type", "Sensor_num", "Distance [mm]"])
for st in dff["Source_type"].unique():
    cols = ["Bx [mT]", "Bz [mT]"]
    dff.loc[dff["Source_type"] == st, cols] -= df.loc[
        df["Source_type"] == "FEM-fine", cols
    ].values

fig = px.line(
    dff,
    x="Distance [mm]",
    y=["Bx [mT]", "Bz [mT]"],
    facet_col="Sensor_num",
    facet_row="variable",
    color="Source_type",
    height=600,
    title='Magpylib & Magpylib+Demag (diff vs FEM-fine)',
)
fig.update_yaxes(matches=None, showticklabels=True)
```
