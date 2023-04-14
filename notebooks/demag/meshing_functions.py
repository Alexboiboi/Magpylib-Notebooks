from itertools import product

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R


def apportion_triple(triple, min_val=1, max_iter=30):
    """Apportion values of a triple, so that the minimum value `min_val` is respected
    and the product of all values remains the same.
    Example: apportion_triple([1,2,50], min_val=3) -> [ 2.99999999  3.         11.11111113]
    """
    triple = np.abs(np.array(triple, dtype=float))
    count = 0
    while any(n < min_val for n in triple) and count < max_iter:
        count += 1
        amin, amax = triple.argmin(), triple.argmax()
        factor = min_val / triple[amin]
        if triple[amax] >= factor * min_val:
            triple /= factor**0.5
            triple[amin] *= factor**1.5
    return triple


def cells_from_dimension(
    dim,
    target_elems,
    min_val=1,
    strict_max=False,
    parity=None,
):
    """Divide a dimension triple with a target scalar of elements, while apportioning the number
    of elements based on the dimension proportions. The resulting divisions are the closest to
    cubes.

    Parameters
    ----------
    dim: array_like of length 3
        Dimensions of the object to be divided.
    target_elems: int,
        Total number of elements as target for the procedure. Actual final number is likely to
        differ.
    min_val: int
        Minimum value of the number of divisions per dimension.
    strict_max: bool
        If `True`, the `target_elem` value becomes a strict maximum and the product of the
        resulting triple will be strictly smaller than the target.
    parity: {None, 'odd', 'even'}
        All elements of the resulting triple will match the given parity. If `None`, no parity
        check is performed.

    Returns
    -------
    numpy.ndarray of length 3
        array corresponding of the number of divisions for each dimension

    Examples
    --------
    >>> cells_from_dimension([1, 2, 6], 926, parity=None, strict_max=True)
    [ 4  9 25]  # Actual total: 900
    >>> cells_from_dimension([1, 2, 6], 926, parity=None, strict_max=False)
    [ 4  9 26]  # Actual total: 936
    >>> cells_from_dimension([1, 2, 6], 926, parity='odd', strict_max=True)
    [ 3 11 27]  # Actual total: 891
    >>> cells_from_dimension([1, 2, 6], 926, parity='odd', strict_max=False)
    [ 5  7 27]  # Actual total: 945
    >>> cells_from_dimension([1, 2, 6], 926, parity='even', strict_max=True)
    [ 4  8 26]  # Actual total: 832
    >>> cells_from_dimension([1, 2, 6], 926, parity='even', strict_max=False)
    [ 4 10 24]  # Actual total: 960
    """
    elems = np.prod(target_elems)  # in case target_elems is an iterable

    # define parity functions
    if parity == "odd":
        funcs = [
            lambda x, add=add, fn=fn: int(2 * fn(x / 2) + add)
            for add in (-1, 1)
            for fn in (np.ceil, np.floor)
        ]
    elif parity == "even":
        funcs = [lambda x, fn=fn: int(2 * fn(x / 2)) for fn in (np.ceil, np.floor)]
    else:
        funcs = [np.ceil, np.floor]

    # make sure the number of elements is sufficient
    elems = max(min_val**3, elems)

    # float estimate of the elements while product=target_elems and proportions are kept
    x, y, z = np.abs(dim)
    a = x ** (2 / 3) * (elems / y / z) ** (1 / 3)
    b = y ** (2 / 3) * (elems / x / z) ** (1 / 3)
    c = z ** (2 / 3) * (elems / x / y) ** (1 / 3)
    a, b, c = apportion_triple((a, b, c), min_val=min_val)
    epsilon = elems
    # run all combinations of rounding methods, including parity matching to find the closest
    # triple with the target_elems constrain
    result = [funcs[0](k) for k in (a, b, c)]  # first guess
    for funcs in product(*[funcs] * 3):
        res = [f(k) for f, k in zip(funcs, (a, b, c))]
        epsilon_new = elems - np.prod(res)
        if np.abs(epsilon_new) <= epsilon and all(r >= min_val for r in res):
            if not strict_max or epsilon_new >= 0:
                epsilon = np.abs(epsilon_new)
                result = res
    return np.array(result).astype(int)


def mesh_Cuboid(cuboid, target_elems, verbose=False, **kwargs):
    """
    Split Magpylib cuboid up into small cuboid cells

    Parameters
    ----------
    cuboid: magpylib.magnet.Cuboid object
        Input object to be discretized
    target_elems: triple of int or int
        Target number of cells. If `target_elems` is a triple of integers, the number of divisions
        corresponds respectively to the x,y,z components of the unrotated cuboid in the global CS.
        If an integer, the number of divisions is apportioned proportionally to the cuboid
        dimensions. The resulting meshing cuboid cells are then the closest to cubes as possible.
    verbose: bool
        If True, prints out meshing information

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells
    """
    # TODO: make function compatible with paths
    # load cuboid properties
    pos = cuboid.position
    rot = cuboid.orientation
    dim = cuboid.dimension
    mag = cuboid.magnetization

    if np.isscalar(target_elems):
        nnn = cells_from_dimension(dim, target_elems)
    else:
        nnn = target_elems
    elems = np.prod(nnn)
    if verbose:
        print(
            f"Meshing Cuboid with {nnn[0]}x{nnn[1]}x{nnn[2]}={elems}"
            f"elements (target={target_elems})"
        )

    # secure input type
    nnn = np.array(nnn, dtype=int)

    # new dimension
    new_dim = dim / nnn

    # inside position grid
    xs, ys, zs = [
        np.linspace(d / 2 * (1 / n - 1), d / 2 * (1 - 1 / n), n)
        for d, n in zip(dim, nnn)
    ]
    grid = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    grid = rot.apply(grid) + pos

    # create cells as magpylib objects and return Collection
    cells = [magpy.magnet.Cuboid(mag, new_dim, pp, rot) for pp in grid]

    return magpy.Collection(cells, **kwargs)


def mesh_cuboids_with_cuboids(obj, target_elems, inplace=False):
    """
    Mesh a Cuboid or Collection of Cuboids with a target number of elements.

    Parameters
    ----------
    obj : magpy.magnet.Cuboid or magpy.Collection
        The Cuboid or Collection to mesh.
    target_elems : int
        The target number of elements for the mesh.
    inplace : bool, optional
        If True, perform the mesh operation in-place, modifying the original
        object. If False, create a new object with the mesh operation applied.
        Default is False.

    Returns
    -------
    obj : magpy.magnet.Cuboid or magpy.Collection
        The meshed Cuboid or Collection.

    Notes
    -----
    This function recursively processes any child objects within the input
    Collection, performing the mesh operation on each.
    """
    if isinstance(obj, magpy.magnet.Cuboid):
        label = obj.style.label
        xi = getattr(obj, "xi", None)
        cuboid_meshed = mesh_Cuboid(obj, target_elems, style_label=label)
        if inplace:
            parent = obj.parent
            obj.parent = None
            parent.add(cuboid_meshed)
        if xi is not None:
            cuboid_meshed.xi = xi
        obj = cuboid_meshed
    elif isinstance(obj, magpy.Collection):
        if not inplace:
            obj = obj.copy()
        children = list(
            obj.children
        )  # otherwise children list is changed while looping!!
        for child in children:
            mesh_cuboids_with_cuboids(child, target_elems, inplace=True)
    return obj


def mesh_Cylinder(cylinder, target_elems, verbose=False):
    """
    Split `Cylinder` or `CylinderSegment` up into small cylindrical or cylinder segment cells.
    In case of the cylinder, the middle cells are cylinders, all other being cylinder segments

    Parameters
    ----------
    cylinder: `magpylib.magnet.Cylinder` or  `magpylib.magnet.CylinderSegment` object
        Input object to be discretized
    target_elems: int
        Target number of cells. If `target_elems` is a triple of integers, the number of divisions
        corresponds respectively to the divisions along the circumference, over the radius and
        over the height. If an integer, the number of divisions is apportioned proportionally to
        the cylinder (or cylinder segment) dimensions. The resulting meshing cylinder segment
        cells are then the closest to cubes as possible.
    verbose: bool
        If True, prints out meshing information

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cylinder and CylinderSegment cells"""
    if isinstance(cylinder, magpy.magnet.CylinderSegment):
        r1, r2, h, phi1, phi2 = cylinder.dimension
    elif isinstance(cylinder, magpy.magnet.Cylinder):
        r1, r2, h, phi1, phi2 = (
            0,
            cylinder.dimension[0] / 2,
            cylinder.dimension[1],
            0,
            360,
        )
    else:
        raise TypeError("Input must be a Cylinder or CylinderSegment")

    pos = cylinder._position
    rot = cylinder._orientation
    mag = cylinder.magnetization
    al = (r2 + r1) * 3.14 * (phi2 - phi1) / 360  # arclen = D*pi*arcratio
    dim = al, r2 - r1, h

    # "unroll" the cylinder and distribute the target number of elemens along the circumference,
    # radius and height.
    if np.isscalar(target_elems):
        nphi, nr, nh = cells_from_dimension(dim, target_elems)
    else:
        nphi, nr, nh = target_elems
    elems = np.prod([nphi, nr, nh])
    if verbose:
        print(
            f"Meshing CylinderSegement with {nphi}x{nr}x{nh}={elems}"
            f" elements (target={target_elems})"
        )
    r = np.linspace(r1, r2, nr + 1)
    dh = h / nh
    cyl_segs = []
    for r_ind in range(nr):
        # redistribute number divisions proportionally to the radius
        nphi_r = max(1, int(r[r_ind + 1] / ((r1 + r2) / 2) * nphi))
        phi = np.linspace(phi1, phi2, nphi_r + 1)
        for h_ind in range(nh):
            pos_h = dh * h_ind - h / 2 + dh / 2
            # use a cylinder for the innermost cells, cylinder segment otherwise
            if r[r_ind] == 0 and phi2 - phi1 == 360:
                dimension = r[r_ind + 1] * 2, dh
                cs = magpy.magnet.Cylinder(
                    magnetization=mag, dimension=dimension, position=(0, 0, pos_h)
                )
                cyl_segs.append(cs)
            else:
                for phi_ind in range(nphi_r):
                    dimension = (
                        r[r_ind],
                        r[r_ind + 1],
                        dh,
                        phi[phi_ind],
                        phi[phi_ind + 1],
                    )
                    cs = magpy.magnet.CylinderSegment(
                        magnetization=mag, dimension=dimension, position=(0, 0, pos_h)
                    )
                    cyl_segs.append(cs)
    return magpy.Collection(cyl_segs).rotate(rot, anchor=0, start=0).move(pos, start=0)


def get_volume(obj, return_containing_cube_edge=False):
    """Return object volume in mm^3. The `containting_cube_edge` is the mininum side length of an
    unrotated cube centered at the origin containing the object.
    """
    if obj.__class__.__name__ == "Cuboid":
        dim = obj.dimension
        vol = dim[0] * dim[1] * dim[2]
        containing_cube_edge = max(obj.dimension)
    elif obj.__class__.__name__ == "Cylinder":
        d, h = obj.dimension
        vol = h * np.pi * (d / 2) ** 2
        containing_cube_edge = max(d, h)
    elif obj.__class__.__name__ == "CylinderSegment":
        r1, r2, h, phi1, phi2 = obj.dimension
        vol = h * np.pi * (r2**2 - r1**2) * (phi2 - phi1) / 360
        containing_cube_edge = max(h, 2 * r2)
    elif obj.__class__.__name__ == "Sphere":
        vol = 4 / 3 * np.pi * (obj.diameter / 2) ** 3
        containing_cube_edge = obj.diameter
    else:
        raise TypeError("Unsupported object type for volume calculation")
    if return_containing_cube_edge:
        return vol, containing_cube_edge
    return vol


def mask_inside_Cuboid(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a Cuboid"""
    a, b, c = obj.dimension / 2
    x, y, z = positions.T
    mx = (abs(x) - a) < tolerance * a
    my = (abs(y) - b) < tolerance * b
    mz = (abs(z) - c) < tolerance * c
    return mx & my & mz


def mask_inside_Cylinder(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a Cylinder"""
    # transform to Cy CS
    x, y, z = positions.T
    r, phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    r0, z0 = obj.dimension.T / 2

    # scale invariance (make dimensionless)
    r = np.copy(r / r0)
    z = np.copy(z / r0)
    z0 = np.copy(z0 / r0)

    m2 = np.abs(z) <= z0  # in-between top and bottom plane
    m3 = r <= 1  # inside Cylinder hull plane

    return m2 & m3


def mask_inside_Sphere(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a Sphere"""
    x, y, z = np.copy(positions.T)
    r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
    r0 = abs(obj.diameter) / 2
    return r - r0 < 0


def mask_inside_CylinderSegment(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a CylinderSegment"""
    close = lambda arg1, arg2: np.isclose(arg1, arg2, rtol=tolerance, atol=tolerance)
    r1, r2, h, phi1, phi2 = obj.dimension.T
    r1 = abs(r1)
    r2 = abs(r2)
    h = abs(h)
    z1, z2 = -h / 2, h / 2

    # transform dim deg->rad
    phi1 = phi1 / 180 * np.pi
    phi2 = phi2 / 180 * np.pi
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T

    # transform obs_pos to Cy CS --------------------------------------------
    x, y, z = positions.T
    r, phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    pos_obs_cy = np.concatenate(((r,), (phi,), (z,)), axis=0).T

    # determine when points lie inside and on surface of magnet -------------

    # phip1 in [-2pi,0], phio2 in [0,2pi]
    phio1 = phi
    phio2 = phi - np.sign(phi) * 2 * np.pi

    # phi=phi1, phi=phi2
    mask_phi1 = close(phio1, phi1) | close(phio2, phi1)
    mask_phi2 = close(phio1, phi2) | close(phio2, phi2)

    # r, phi ,z lies in-between, avoid numerical fluctuations (e.g. due to rotations) by including tolerance
    mask_r_in = (r1 - tolerance < r) & (r < r2 + tolerance)
    mask_phi_in = (np.sign(phio1 - phi1) != np.sign(phio1 - phi2)) | (
        np.sign(phio2 - phi1) != np.sign(phio2 - phi2)
    )
    mask_z_in = (z1 - tolerance < z) & (z < z2 + tolerance)

    # inside
    mask_inside = mask_r_in & mask_phi_in & mask_z_in
    return mask_inside


def mask_inside(obj, positions, tolerance=1e-14):
    """Return mask of provided positions inside a Magpylib object"""
    mask_inside_funcs = {
        "Cuboid": mask_inside_Cuboid,
        "Cylinder": mask_inside_Cylinder,
        "Sphere": mask_inside_Sphere,
        "CylinderSegment": mask_inside_CylinderSegment,
    }
    func = mask_inside_funcs.get(obj.__class__.__name__, None)
    if func is None:
        raise TypeError("Unsupported object type for inside masking")
    return func(obj, positions, tolerance)


def mesh_with_cubes(obj, target_elems, strict_inside=True):
    """
    Split-up a Magpylib magnet into a regular grid of identical cells. A grid of identical cube
    cells and containing the object is created. Only the cells with their barycenter inside the
    original object are kept.

    Parameters
    ----------
    obj: `magpylib.magnet` object
        Input object to be discretized
    target_elems: int
        Target number of cells
    strict inside: bool
        If True, also filters out the cells with vertices outside the object boundaries

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cylinder and CylinderSegment cells"""
    vol, containing_cube_edge = get_volume(obj, return_containing_cube_edge=True)
    vol_ratio = (containing_cube_edge**3) / vol

    grid_elems = [int((vol_ratio * target_elems) ** (1 / 3))] * 3
    grid_dim = [containing_cube_edge] * 3

    slices = [slice(-d / 2, d / 2, N * 1j) for d, N in zip(grid_dim, grid_elems)]
    grid = np.mgrid[slices].reshape(len(slices), -1).T
    grid = grid[mask_inside(obj, grid, tolerance=1e-14)]
    cube_cell_dim = np.array([containing_cube_edge / (grid_elems[0] - 1)] * 3)
    if strict_inside:
        elemgrid = np.array(
            list(product(*[[-cube_cell_dim[0] / 2, cube_cell_dim[0] / 2]] * 3))
        )
        cube_grid = np.array([elemgrid + pos for pos in grid])
        pos_inside_strict_mask = np.all(
            mask_inside(obj, cube_grid.reshape(-1, 3)).reshape(cube_grid.shape[:-1]),
            axis=1,
        )
        grid = grid[pos_inside_strict_mask]
        if grid.size == 0:
            raise ValueError("No cuboids left with strict-inside method")
    cube_poss = obj.orientation.apply(grid) + obj.position

    obj_list = [
        magpy.magnet.Cuboid(
            magnetization=obj.magnetization,
            dimension=cube_cell_dim,
            position=pos,
            orientation=obj.orientation,
        )
        for pos in cube_poss
    ]
    return magpy.Collection(obj_list)


def mesh_thin_CylinderSegment_with_cuboids(cyl_seg, target_elems, thin_ratio_limit=10):
    """
    Split-up a Magpylib thin-walled cylinder segment into cuboid cells. Over the thickness, only
    one layer of cells is used.

    Parameters
    ----------
    cyl_seg: `magpylib.magnet.CylinderSegment` object
        CylinderSegment object to be discretized
    target_elems: int or tuple of int,
        If `target_elems` is a  tuple of integers, the cylinder segment is respectively divided
        over circumference and height, if an integer, divisions are infered to build cuboids with
        close to squared faces.
    strict thin_ratio_limit: positive number,
        Sets the r2/(r2-r1) limit to be considered as thin-walled, r1 being the inner radius

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells"""

    r1, r2, h, phi1, phi2 = cyl_seg.dimension
    if thin_ratio_limit > r2 / (r2 - r1):
        raise ValueError(
            "This meshing function is intended for thin-walled CylinderSegment objects"
            f" of radii-ratio r2/(r2-r1)>{thin_ratio_limit}, r1 being the inner radius."
            f"\nInstead r2/(r2-r1)={r2 / (r2 - r1)}"
        )
    # distribute elements -> targeting thin close-to-square surface cells
    circumf = 2 * np.pi * r1
    if np.isscalar(target_elems):
        nphi = int(np.round((target_elems * circumf / h) ** 0.5))
        nh = int(np.round(target_elems / nphi))
    else:
        nphi, nh = target_elems
    dh = h / nh
    dphi = 2 * np.pi / nphi * (phi2 - phi1) / 360
    a, b, c = r2 - r1, 2 * r1 * np.sin(dphi / 2), dh  # cuboids edge sizes
    x0 = r1 * np.cos(dphi / 2) + a / 2
    phi_vec = np.linspace(
        np.deg2rad(phi1) + dphi / 2, np.deg2rad(phi2) - dphi / 2, nphi
    )
    poss = np.array([x0 * np.cos(phi_vec), x0 * np.sin(phi_vec), np.zeros(nphi)]).T
    rots = R.from_euler("z", phi_vec)
    cuboids = [
        magpy.magnet.Cuboid(
            cyl_seg.magnetization, (a, b, c), pos + np.array([0, 0, z]), orient
        )
        for pos, orient in zip(poss, rots)
        for z in np.linspace(-h / 2 + dh / 2, h / 2 - dh / 2, nh)
    ]
    col = magpy.Collection(cuboids)
    col.orientation = cyl_seg.orientation
    col.position = cyl_seg.position
    return col


def slice_Cuboid(cuboid, shift=0.5, axis="z", **kwargs):
    """
    Slice a cuboid magnet along a specified axis and return a collection of the resulting parts.

    Parameters
    ----------
    cuboid : magpy.magnet.Cuboid
        The cuboid to be sliced.
    shift : float, optional, default=0.5
        The relative position of the slice along the specified axis, ranging from
        0 to 1 (exclusive).
    axis : {'x', 'y', 'z'}, optional, default='z'
        The axis along which to slice the cuboid.
    **kwargs
        Additional keyword arguments to pass to the magpy.Collection constructor.

    Returns
    -------
    coll : magpy.Collection
        A collection of the resulting parts after slicing the cuboid.

    Raises
    ------
    ValueError
        If the shift value is not between 0 and 1 (exclusive).
    """
    if not 0 < shift < 1:
        raise ValueError("Shift must be between 0 and 1 (exclusive)")
    dim0 = cuboid.dimension
    mag0 = cuboid.magnetization
    xi = getattr(magpy, "xi", None)
    ind = "xyz".index(axis)
    dim_k = cuboid.dimension[ind]
    dims_k = dim_k * (1 - shift), dim_k * (shift)
    shift_k = (dim_k - dims_k[0]) / 2, -(dim_k - dims_k[1]) / 2
    children = []
    for d, s in zip(dims_k, shift_k):
        dimension = dim0.copy()
        dimension[ind] = d
        position = np.array([0, 0, 0], dtype=float)
        position[ind] = s
        child = magpy.magnet.Cuboid(
            magnetization=mag0, dimension=dimension, position=position
        )
        if xi is not None:
            child.xi = xi
        children.append(child)
    coll = magpy.Collection(children, **kwargs)
    coll.orientation = cuboid.orientation
    coll.position = cuboid.position
    return coll
