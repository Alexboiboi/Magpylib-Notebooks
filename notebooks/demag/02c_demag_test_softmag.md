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

```{code-cell}
%load_ext autoreload
%autoreload 2
```

```{code-cell}
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

```{code-cell}
kwargs = dict(
    max_passes=10, init_refine_factor=100, refine_factor=8, max_dist=1, mag_diff_thresh=2000, max_elems=2000
)
coll = apply_demag_with_refinement(
    collection=COLL0,
    inplace=False,
    **kwargs,
)
```

```{code-cell}
from matplotlib import cm
import plotly.graph_objects as go

mag_linear_lim = 1100 # mT
color_map = cm.get_cmap('viridis', 20)
mags = np.linalg.norm([src.magnetization for src in coll.sources_all], axis=1)
mags_normed = (mags-min(mags))/ (mag_linear_lim-min(mags)) # normalize
colors = color_map(mags_normed)
for src,mag,color in zip(coll.sources_all, mags, colors):
    if mag>mag_linear_lim:
        src.style.magnetization.color.middle = 'grey'
    else:
        src.style.magnetization.color.middle = tuple(color)
        
fig = go.Figure()
magpy.show(coll, canvas=fig)
fig.update_layout(height=600)
fig
```
