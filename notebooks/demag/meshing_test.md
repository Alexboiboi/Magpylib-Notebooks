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
import plotly.graph_objects as go
from meshing_functions import mesh_cuboid, mesh_cylinder, mesh_with_cubes
```

# Exact meshing

```{code-cell} ipython3
cyl = magpy.magnet.CylinderSegment(
    magnetization=(0, 0, 1000), dimension=(0, 2, 1, 0, 360)
)
cyl.move((0, 0, 0))
#cyl.rotate_from_angax(45, "y")
cyl.style.magnetization.show = False
cyl.style.opacity = 0.5

mesh = mesh_cylinder(cyl, 20)
mesh.set_children_styles(magnetization_show=False)

epsilon = 1e-5
sens = magpy.Sensor().move(np.linspace((-5,epsilon,epsilon),(5,epsilon,epsilon), 101), start=0)

magpy.show(sens, cyl, *mesh, backend="plotly")
mesh
```

```{code-cell} ipython3
fig = go.FigureWidget()
fig.add_scatter(y=sens.getB(cyl).T[2], name='B-field single')
fig.add_scatter(y=sens.getB(mesh).T[2], name='B-field mesh')
#fig.add_scatter(y=sens.getH(cyl).T[2], name='H-field single')
#fig.add_scatter(y=sens.getH(mesh).T[2], name='H-field mesh')
fig.show()
```

# Mesh with cuboids

```{code-cell} ipython3
obj = magpy.magnet.CylinderSegment((100, 200, 300), (10, 30, 50, 0, 360)).move(
    (2.2, 0, 0)
)
# obj = magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1))
# obj = magpy.magnet.Cylinder((0, 0, 1), (1, 1))
# obj = magpy.magnet.Sphere((0, 0, 1), 1)

# obj.move([10,0,0]).rotate_from_angax(75, (1,5,6), anchor=0)

meshed_obj = mesh_with_cubes(obj, target_elems=1100, strict_inside=True)
meshed_obj
```

```{code-cell} ipython3
import plotly.graph_objects as go

fig = go.Figure()
magpy.show(
    # obj,
    meshed_obj,
    canvas=fig,
    # backend="matplotlib",
    backend="plotly",
)
fig
```

```{code-cell} ipython3

```
