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
import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
from meshing_functions import (
    mesh_Cuboid,
    mesh_Cylinder,
    mesh_thin_CylinderSegment_with_cuboids,
    mesh_with_cubes,
)

magpy.defaults.display.backend = "plotly"
magpy.defaults.display.style.magnet.magnetization.show = False
```

# Exact meshing

+++

## Cuboid

```{raw-cell}
cube = magpy.magnet.Cuboid(magnetization=(0, 0, 1000), dimension=(4, 3, 2)).move(
    (-1.5, 0, 0)
)

target_elems = (2, 3, 4)
print("Explicit mesh division, target_elems: ", target_elems)
meshed_cube1 = mesh_Cuboid(cube, target_elems)
magpy.show(*meshed_cube1)

target_elems = 27
print("Implicit mesh division, target_elems: ", target_elems)
meshed_cube2 = mesh_Cuboid(cube, target_elems)
magpy.show(*meshed_cube2)
```

## Cylinder/CylinderSegment

```{raw-cell}
cyl = magpy.magnet.CylinderSegment(
    magnetization=(0, 0, 1000), dimension=(1, 2, 1, 0, 360)
)
cyl.move((0, 0, 0))
cyl.rotate_from_angax(45, "y", anchor=0)


epsilon = 1e-5
sens = (
    magpy.Sensor()
    .move(np.linspace((-3, epsilon, epsilon), (3, epsilon, epsilon), 101), start=0)
    .rotate_from_angax(45, "y", anchor=0)
)
target_elems = 50
mesh1 = mesh_Cylinder(cyl, target_elems)
print("Implicit mesh division, target_elems: ", target_elems)
magpy.show(*mesh1, sens)

target_elems = (4, 1, 5)
mesh2 = mesh_Cylinder(cyl, target_elems)
print("Explicit mesh division, target_elems: ", target_elems)
magpy.show(*mesh2, sens)

fig = go.Figure(layout_title="Field over sensor path")
fig.add_scatter(y=sens.getB(cyl).T[2], name="B-field (Full)")
fig.add_scatter(y=sens.getB(mesh1).T[2], name="B-field (Meshed-1)")
fig.add_scatter(y=sens.getB(mesh2).T[2], name="B-field (Meshed-2)")
# fig.add_scatter(y=sens.getH(cyl).T[2], name='H-field single')
# fig.add_scatter(y=sens.getH(mesh).T[2], name='H-field mesh')
fig.show()
```

# Approximate meshing

## Mesh with cubes

```{raw-cell}
objs = [
    magpy.magnet.CylinderSegment((100, 200, 300), (10, 30, 50, 0, 360)).move(
        (2.2, 0, 0)
    ),
    magpy.magnet.Cuboid((1, 0, 0), (1, 1, 1)),
    magpy.magnet.Cylinder((0, 0, 1), (1, 1)),
    magpy.magnet.Sphere((0, 0, 1), 1),
]
for obj in objs:
    obj.style.opacity = 0.5
    # obj.move([10,0,0]).rotate_from_angax(75, (1,5,6), anchor=0)
    mesh1 = mesh_with_cubes(obj, 200, strict_inside=False)
    mesh2 = mesh_with_cubes(obj, 200, strict_inside=True)

    print(f"{obj._object_type} meshed ({len(mesh1)} cubes)")
    magpy.show(obj, *mesh1)
    print(f"{obj._object_type} stric-inside ({len(mesh2)} cubes)")
    magpy.show(obj, *mesh2)
```

## Mesh thin cylinder segment with cuboids

```{code-cell} ipython3
cyl_seg = magpy.magnet.CylinderSegment((100, 200, 300), (29, 30, 50, 0, 360)).move(
    (2.2, 0, 0)
)

target_elems = (6, 1)
print("Explicit mesh division, target_elems: ", target_elems)
cyl_seg_mesh1 = mesh_thin_CylinderSegment_with_cuboids(cyl_seg, target_elems)
magpy.show(*cyl_seg_mesh1)

target_elems = 27
print("Implicit mesh division, target_elems: ", target_elems)
cyl_seg_mesh2 = mesh_thin_CylinderSegment_with_cuboids(cyl_seg, target_elems)
magpy.show(*cyl_seg_mesh2)
```
