---
jupytext:
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
import plotly.graph_objects as go
from demag_functions import mesh_cylinder
```

```{code-cell} ipython3
cyl = magpy.magnet.CylinderSegment(
    magnetization=(0, 0, 1000), dimension=(0, 2, 1, 0, 360)
)
cyl.move((0, 0, 0))
#cyl.rotate_from_angax(45, "y")
cyl.style.magnetization.show = False
cyl.style.opacity = 0.5

mesh = mesh_cylinder(cyl, 100)
mesh.set_children_styles(magnetization_show=False)

sens = magpy.Sensor().move(np.linspace((-5,0,0),(5,0,0), 101), start=0)

magpy.show(sens, cyl, *mesh, backend="plotly")
mesh
```

```{code-cell} ipython3
fig = go.FigureWidget()
fig.add_scatter(y=sens.getB(cyl).T[2], name='B-field single')
fig.add_scatter(y=sens.getB(mesh).T[2], name='B-field mesh')
fig.add_scatter(y=sens.getH(cyl).T[2], name='H-field single')
fig.add_scatter(y=sens.getH(mesh).T[2], name='H-field mesh')
fig.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
