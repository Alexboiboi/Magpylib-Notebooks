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
import numpy as np
import pandas as pd
import plotly.express as px
import magpylib as magpy

from demag_functions import apply_demag
from meshing_functions import mesh_Cylinder

magpy.defaults.display.backend =  'plotly'

magnet_diameter = 40
magnet_height = 2.2

temperature = 25
Brem_nominal=1130.0
Brem_tc_perc=-0.115
Brem_at_temperature = Brem_nominal * (1 + (temperature - 25) * Brem_tc_perc / 100)

cyl_fn = lambda: magpy.magnet.Cylinder(
        (Brem_at_temperature, 0, 0),
        (magnet_diameter, magnet_height),
    )
cyl = cyl_fn()
def create_meshed_cylinder(xi=0.05, elems=1):
    cyl = cyl_fn()
    cyl_mesh = mesh_Cylinder(cyl, elems)
    cyl_mesh.xi = xi
    cyl_mesh.style.label = f"Meshed Cylinder ({elems=})"
    apply_demag(cyl_mesh)
    return cyl_mesh
h=9
sens = magpy.Sensor().move(np.linspace((-2,0,h),(2,0,h),50), start=0)
```

```{code-cell} ipython3
B_list = [magpy.getB(cyl, sens, output='dataframe')]
for elems in [10, 300]:#, 20, 50, 100, 200, 300]:
    cyl_mesh = create_meshed_cylinder(elems=elems)
    B  = magpy.getB(cyl_mesh, sens, output='dataframe')
    B_list.append(B)
B = pd.concat(B_list)
```

```{code-cell} ipython3
magpy.show(cyl_mesh, sens)
```

```{code-cell} ipython3
B['Bmag'] = (B['Bx']**2 + B['By']**2 + B['Bz']**2)**0.5
```

```{code-cell} ipython3
fig = px.line(B, color='source', x='path', y=['Bx','By', 'Bz', 'Bmag'], facet_col='variable')
fig.update_yaxes(matches=None, showticklabels=True)
```
