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
import magpylib as magpy
import numpy as np
import pandas as pd
import plotly.express as px
from demag_functions import apply_demag, apply_demag_with_refinement
from meshing_functions import mesh_Cuboid

magpy.defaults.display.backend = "plotly"

elems = 200  # mesh factor

# hard magnets
mag1 = np.array((0, 0, 1000))
cube_side_len = 5
dim1 = np.array([cube_side_len] * 3)
pos1 = np.array((-cube_side_len, 0, 0))
cube1 = magpy.magnet.Cuboid(mag1, dim1, pos1)
cube1.xi = 0.0

cube2 = magpy.magnet.Cuboid(-mag1, dim1, -pos1)
cube2.xi = 0.0


# soft magnet
plate_thickness = 1
plate_len = cube_side_len * 3
plate = magpy.magnet.Cuboid(
    (0, 0, 0),
    (plate_len, cube_side_len, plate_thickness),
    (0, 0, -cube_side_len / 2 - plate_thickness / 2),
)
plate.xi = 3999

# super collection
COLL0 = magpy.Collection(cube1, cube2, plate)
# magpy.show(cube1, cube2, sensors)

# add sensors
sensors = [
    magpy.Sensor(
        position=np.linspace((-15, 0, z), (15, 0, z), 1001),
    )
    for z in [cube_side_len * 1.5]
]
```

```{raw-cell}
magpy.show(*COLL0, sensors, style_magnetization_show=False)
```

```{code-cell} ipython3
kwargs = dict(
    max_passes=16, init_refine_factor=10, refine_factor=2, max_dist=1, mag_diff_thresh=15, max_elems=1000
)
coll = apply_demag_with_refinement(
    collection=COLL0,
    inplace=False,
    **kwargs,
)
```

```{code-cell} ipython3
#for src in coll.sources_all:
#    src.style.magnetization.show = True if np.linalg.norm(src.magnetization) > 500 else False
import plotly.graph_objects as go

fig = go.Figure()
magpy.show(coll, canvas=fig)
fig.update_layout(height=600)
fig
```
