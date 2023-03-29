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
import glob
import os
import pickle
from datetime import timedelta
from time import perf_counter

import ipywidgets as widgets
import magpylib as magpy
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from atalex._lib.utils import (
    calc_angles_with_compens,
    calc_compens_params,
    calc_dtheta_compensation_on_off,
)
from demag_functions import apply_demag
from magpylib_extras.sensor import VolumeSensor
from magpylib_extras.source import DiscreteSourceBox
from magpylib_extras.utility import stl2mesh3d
from meshing_functions import (
    mesh_Cuboid,
    mesh_Cylinder,
    mesh_thin_CylinderSegment_with_cuboids,
)
from scipy.spatial.transform.rotation import Rotation

ANSYS_file_params = {
    "usecols": [0, 1, 2, 4, 5, 6],
    "names": ["x", "y", "z", "Bx", "By", "Bz"],
    "factors": [1000, 1000, 1000, 1000, 1000, 1000],
    "delimiter": " ",
    "skiprows": 2,
    "dtype": "float64",
    "na_values": ["Nan"],
    "fill_value": np.nan,
}

magpy.defaults.display.backend = "plotly"
```

```{code-cell} ipython3
class HalbachCylinder(magpy.Collection):
    def __init__(self, num_of_mags=16, **kwargs):
        super().__init__()
        self.update(num_of_mags=num_of_mags, **kwargs)
        self.inner_radius = lambda r, n: r * (1 - np.sqrt(2) * self.xi_up(n))
        self.outer_radius = lambda r, n: r * (1 + np.sqrt(2) * self.xi_up(n))

    @staticmethod
    def xi_up(n):
        # https://onlinelibrary.wiley.com/doi/epdf/10.1002/cmr.a.20165  page 213
        a = (
            np.cos(2 * np.pi / n)
            - np.sin(2 * np.pi / n)
            - np.sqrt(2) * np.sin(np.pi / 4 - 4 * np.pi / n)
        )
        b = 2 * np.cos(np.pi / 4 - 4 * np.pi / n) + np.sqrt(2)
        return a / b

    def update(
        self,
        num_of_mags=16,
        middle_radius=28,
        mag_len=50,
        mag_side_len=3.5,
        mag_side_x=0.0,
        mag_side_y=0.0,
        mag_tilt_deg=0.0,
        Brem=1040.0,
        Brem_perc_var=0,
        site_num=None,
    ):
        mag_num_diff = num_of_mags - len(self.sources)
        if mag_num_diff > 0:
            self.add([magpy.magnet.Cuboid() for _ in range(mag_num_diff)])
        elif mag_num_diff < 0:
            self.sources = self.sources[:mag_num_diff]
        sources = self.sources
        mag_vert = (
            2 * middle_radius * self.xi_up(num_of_mags)
            if mag_side_len == 0
            else mag_side_len
        )
        if mag_side_x == 0 and mag_side_y == 0:
            mag_side_x = mag_side_y = mag_vert
        alpha_deg_list = np.linspace(0, 360, len(sources), endpoint=False)
        Brems = np.array([Brem] * len(sources))
        if Brem_perc_var != 0:
            Brems += Brem_perc_var * np.random.uniform(
                low=-1, high=1, size=len(sources)
            )
        for ind, (magnet, alpha_deg, Brem) in enumerate(
            zip(sources, alpha_deg_list, Brems)
        ):
            magnet.style.label = f"HB_{ind+1:02d}" + (
                "" if site_num is None else "_{site_num:02d}"
            )
            magnet.reset_path()
            magnet.rotate_from_angax(angle=alpha_deg, axis=(0, 0, 1))
            magnet.position = np.array([middle_radius, 0, 0])
            magnet.rotate_from_angax(
                angle=-mag_tilt_deg,
                axis=(1, 0, 0),
                anchor=(middle_radius, 0, mag_len / 2),
            )
            magnet.rotate_from_angax(angle=alpha_deg, axis=(0, 0, 1), anchor=(0, 0, 0))
            magnet.dimension = np.array([mag_side_x, mag_side_y, mag_len])
            magnet.magnetization = np.array([Brem, 0.0, 0.0])
```

```{code-cell} ipython3
def create_pill_grid(dist, rows=1, cols=1, recenter=False):
    pos = []
    for r in range(rows):
        for c in range(cols):
            pos.append(np.array([c * dist, r * dist, 0]))
    pos = np.array(pos)
    if recenter:
        pos = pos - 0.5 * (np.max(pos, axis=0) - np.min(pos, axis=0)).T
    return list(pos)


fparams_ANSYS = ANSYS_file_params.copy()
fparams_ANSYS["skiprows"] = 1
COLORS = tuple(magpy.defaults.display.colorsequence)
out = widgets.Output()


def create_source(filepath, **params):
    return DiscreteSourceBox.from_file(filepath, **params)


def calc_dtheta_deg(B_array, angles):
    res = calc_dtheta_compensation_on_off(B_array.T[0].T, B_array.T[1].T, angles)
    return res[0]


def calc_Barray(
    dataset, airgap=0, Delta=0, zRot=0, yTilt=0, xTilt=0, sensor_dim=0, Nelem=1
):
    angles = dataset.angles
    pill_distance = dataset.pill_distance
    rows = getattr(dataset, "rows", None)
    cols = getattr(dataset, "cols", 1)
    sensor = VolumeSensor(Nelem=[Nelem] * 3, dimension=[sensor_dim] * 3)
    # sensor = magpy.Sensor()
    # rotate/move sensor
    rs = Rotation.from_euler("zyx", [zRot, yTilt, xTilt], degrees=True)
    axis_s = rs.as_rotvec()
    angle_s = np.rad2deg(np.linalg.norm(axis_s))
    if angle_s != 0:
        sensor.rotate_from_angax(angle=angle_s, axis=axis_s, anchor=(0, 0, 0))
    sensor.move((Delta, 0, -airgap))

    if len(dataset.collections) == 1:
        coll = dataset.collections[0]
        if rows is None:
            rows = len(coll)
        grid = create_pill_grid(dist=pill_distance, rows=rows, cols=cols)
        for ind, s in enumerate(coll.children):
            s.position = grid[ind]
            s.orientation = None
            s.rotate_from_angax(angle=angles, axis="z", start=0)
        Barray = sensor.getB(coll)
    else:  # single FEM only
        Barray = np.array([sensor.getB(coll) for coll in dataset.collections])
    if Barray.ndim == 3:
        Barray = Barray.mean(axis=-2)
    return Barray


def show_Bxyz(dsets, reference=None):
    if reference is not None:
        title_suff = " vs " + reference
        dset_ref = [dset for dset in dsets if dset.style.label == reference][0]
    else:
        title_suff = ""
    fig = go.FigureWidget().set_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02
    )
    fig.update_layout(
        title="B-field values while rotating" + title_suff,
        xaxis3_title="nominal angle [deg]",
        yaxis_title="Bx [mT]",
        yaxis2_title="By [mT]",
        yaxis3_title="Bz [mT]",
    )

    def update_fig(**kwargs):
        if reference is not None:
            Barray_ref = calc_Barray(dset_ref, **kwargs)
        with fig.batch_update():
            for j, dset in enumerate(dsets):
                Barray = calc_Barray(dset, **kwargs)
                if reference is not None:
                    Barray -= Barray_ref
                for i, k in enumerate("xyz"):
                    trace = getattr(dset, f"trace_B{k}")
                    trace.y = Barray.T[i]

    for j, dset in enumerate(dsets):
        for i, k in enumerate("xyz"):
            fig.add_scatter(
                x=dset.angles,
                name=dset.style.label,
                legendgroup=dset.style.label,
                line_color=COLORS[j],
                row=i + 1,
                col=1,
                showlegend=True if i == 0 else False,
            )
            setattr(dset, f"trace_B{k}", fig.data[-1])
    w = widgets.interactive(
        update_fig,
        airgap=widgets.FloatSlider(min=-4, max=4, step=0.1, value=0),
        Delta=widgets.FloatSlider(min=-2, max=2, step=0.1, value=0),
        xTilt=widgets.FloatSlider(min=-5, max=5, step=0.1, value=0),
        yTilt=widgets.FloatSlider(min=-5, max=5, step=0.1, value=0),
        sensor_dim=widgets.FloatSlider(min=0, max=2, step=0.01, value=0.1),
        Nelem=widgets.IntSlider(min=1, max=10),
    )
    fig_box = widgets.VBox([fig], layout=dict(flex="1"))
    display(widgets.HBox([w, fig_box]))


def show_residuals(datasets):
    def update_fig(**kwargs):
        with fig.batch_update():
            for i, dset in enumerate(datasets):
                Barray = calc_Barray(dset, **kwargs)
                fig.data[i].y = calc_dtheta_deg(Barray, dset.angles)

    fig = go.FigureWidget()
    fig.update_layout(
        title="Crosstalk Method Comparison - Calculated angle error",
        xaxis_title="nominal angle [deg]",
        yaxis_title="dtheta [deg]",
    )

    for i, dset in enumerate(datasets):
        fig.add_scatter(x=dset.angles, line_color=COLORS[i], name=dset.style.label)

    w = widgets.interactive(
        update_fig,
        airgap=widgets.FloatSlider(min=-4, max=4, step=0.1, value=0),
        Delta=widgets.FloatSlider(min=-2, max=2, step=0.1, value=0),
        xTilt=widgets.FloatSlider(min=-5, max=5, step=0.1, value=0),
        yTilt=widgets.FloatSlider(min=-5, max=5, step=0.1, value=0),
        sensor_dim=widgets.FloatSlider(min=0, max=2, step=0.01, value=0.1),
        Nelem=widgets.IntSlider(min=1, max=10),
    )
    fig_box = widgets.VBox([fig], layout=dict(flex="1"))
    display(widgets.HBox([w, fig_box]))
```

```{code-cell} ipython3
def create_meshed_Halbach(
    *,
    shield_mesh_mode,
    angle=0,
    position=(0, 0, 0),
    shield_elems=200,
    cuboids_elems=25,
    **halbach_params,
):
    hc = HalbachCylinder(**halbach_params)
    shield = magpy.magnet.CylinderSegment(
        magnetization=(0, 0, 0), dimension=(32, 32.5, 50, 0, 360)
    )

    hc_meshed = magpy.Collection([mesh_Cuboid(child, cuboids_elems) for child in hc])
    hc_meshed.style.label = "Meshed-Halbach-Cylinder"
    hc_meshed.xi = 0.05
    if shield_mesh_mode == "Cuboid":
        shield_meshed = mesh_thin_CylinderSegment_with_cuboids(shield, shield_elems)
    elif shield_mesh_mode == "CylinderSegment":
        shield_meshed = mesh_Cylinder(shield, shield_elems)
    else:
        raise ValueError(
            """Bad shield_mesh_mode, must be one of '('CylinderSegment','Cuboid')`"""
        )
    shield_meshed.style.label = "Meshed-Cylinder-Shield"
    shield_meshed.xi = 3999
    hc_with_shield_meshed = hc_meshed + shield_meshed
    hc_with_shield_meshed.rotate_from_angax(angle, "z")
    hc_with_shield_meshed.move(position)
    hc_with_shield_meshed.style.label = "Meshed-Halbach-with-shield"
    return hc_with_shield_meshed


def create_demag_Halbach(
    angle=0, pill_distance=75, shield_mesh_mode="Cuboid", **kwargs
):
    hc1 = create_meshed_Halbach(
        angle=angle, shield_mesh_mode=shield_mesh_mode, **kwargs
    )
    hc2 = create_meshed_Halbach(
        angle=angle,
        position=(pill_distance, 0, 0),
        shield_mesh_mode=shield_mesh_mode,
        **kwargs,
    )
    hc_with_shield_meshed = hc1 + hc2
    hc_with_shield_meshed.style.label = f"Meshed-Halbach-with-shield-angle={angle}"
    return hc_with_shield_meshed
```

```{code-cell} ipython3
def create_FEM_dataset(folder, name=None):
    if name is None:
        name = str(folder)
    angles = []
    collections = []
    for file in glob.glob(folder + "/*.fld"):
        angle = float(file.split("__")[1].split("deg")[0].replace("_", "."))
        angles.append(angle)
        coll = magpy.Collection(
            create_source(file, **ANSYS_file_params),
            style_label=str(angle),
        )
        collections.append(coll)
        # print(f"{angle=!r}, {file=!r}")
    dset = magpy.Collection(*collections)
    dset.angles = np.array(sorted(angles))
    dset.style.label = name
    return dset
```

```{code-cell} ipython3
folder = "data/two_pills_with_shield/singlelayer_fine"

datasets = []

datasets.append(
    create_FEM_dataset(folder, name="2Pills_FEM_only_singlelayershield_fine")
)
angles = datasets[0].angles


def to_timedelta_string(tdiff):
    td = timedelta(seconds=tdiff)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h, {minutes}min, {seconds}sec"


load_files = glob.glob("*.pickle")
if load_files:
    for ind, file in enumerate(load_files):
        with open(file, "rb") as handle:
            demag_coll = pickle.load(handle)
            src = demag_coll[0][0]
            lbl = f"{ind:02d} - magnets ({len(src[0])}x): {len(src[0][1])} cuboids, shield: {len(src[1])} cuboids"
            demag_coll.style.label = lbl
        datasets.append(demag_coll)
else:
    colls = []
    start_time = perf_counter()
    for ind, angle in enumerate(angles[:1]):
        print(f"{ind+1}/{len(angles)}   {angle=}°  ")
        src = create_demag_Halbach(
            angle=angle,
            shield_elems=(64, 32),
            cuboids_elems=10,
        )
        apply_demag(src, pairs_matching=True, inplace=True)
        tdiff = perf_counter() - start_time
        remaining_time = to_timedelta_string(tdiff * (len(angles) / (ind + 1) - 1))
        print(
            f"{ind+1}/{len(angles)}   {angle=}°  "
            f"cummulative elapsed time: {to_timedelta_string(tdiff)}   "
            f"estimated remaining time: {remaining_time}"
        )
        colls.append(src)
    demag_coll = magpy.Collection(
        colls, style_label="Meshed-Halbach-with-shield-all_angles"
    )

    datasets.append(demag_coll)

for d in datasets:
    d.angles = angles
    d.pill_distance = 75
```

```{code-cell} ipython3
magpy.show(datasets[-1][0])
```

```{code-cell} ipython3
datasets[-1]
```

```{raw-cell}
import pickle

# Store data (serialize)
with open('demag_hb00.pickle', 'wb') as handle:
    pickle.dump(datasets[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
```

```{code-cell} ipython3
display(out)
show_Bxyz(datasets)
# show_Bxyz(datasets, reference='1Pill_FEM+interpolated_rotation')
show_residuals(datasets)
```

```{raw-cell}
sens = magpy.Sensor(position=(2, 0, 0)).rotate_from_angax(
    np.linspace(0, 360, 100, endpoint=False), "z", start=0, anchor=0
)
fig = go.Figure()
B0 = sens.getB(hc_FEM)
B1 = sens.getB(hc_with_shield_meshed, sumup=True)
B2 = sens.getB(hc_with_shield_meshed_demag)

fig.add_scatter(y=B0[:, 0], name="FEM")
fig.add_scatter(y=B1[:, 0], name="Magpylib-No-Demag")
fig.add_scatter(y=B2[:, 0], name="Magpylib-With-Demag")
fig

fig = go.Figure()
fig.add_scatter(y=B0[:, 0] - B0[:, 0], name="FEM")
fig.add_scatter(y=B1[:, 0] - B0[:, 0], name="Magpylib-No-Demag")
fig.add_scatter(y=B2[:, 0] - B0[:, 0], name="Magpylib-With-Demag")
fig

def dt(B):
    angles = np.linpsace()
    calc_dtheta_compensation_on_off(B.T[0].T, B.T[1].T, angles)
    return dt


fig = go.Figure()
fig.add_scatter(y=dt(B0), name="FEM")
fig.add_scatter(y=dt(B1), name="Magpylib-No-Demag")
fig.add_scatter(y=dt(B2), name="Magpylib-With-Demag")
fig
```
