# pylint: disable=invalid-name, redefined-outer-name
import numpy as np
import magpylib as magpy


def reduce_res(result, min_vert, max_iter=30):
    result = np.abs(np.array(result))
    count = 0
    while any(n < min_vert for n in result) and count < max_iter:
        count += 1
        amin, amax = result.argmin(), result.argmax()
        factor = min_vert / result[amin]
        if result[amax] >= factor * min_vert:
            result /= factor**0.5
            result[amin] *= factor**1.5
    return result


def infer_elems_from_dim(
    dim=(1.0, 1.0, 1.0),
    elems=8,
    min_vert=2,
    strict_max=False,
    parity=False,
    max_iter=30,
    return_elems=False,
):
    elems = np.prod(elems)  # in case elems is an iterable
    if parity == "odd":
        funcs = [
            lambda x: 2 * fn(x / 2) + i for fn in (np.ceil, np.floor) for i in (-1, 1)
        ]
    elif parity == "even":
        funcs = [lambda x: 2 * fn(x / 2) for fn in (np.ceil, np.floor)]
    else:
        funcs = [np.ceil, np.floor]
    from itertools import product

    dim = np.abs(dim)
    elems = max(min_vert**3, elems)
    x, y, z = dim
    a = x ** (2 / 3) * (elems / y / z) ** (1 / 3)
    b = y ** (2 / 3) * (elems / x / z) ** (1 / 3)
    c = z ** (2 / 3) * (elems / x / y) ** (1 / 3)
    epsilon = elems
    a, b, c = reduce_res((a, b, c), min_vert=min_vert, max_iter=max_iter)
    result = [int(k) for k in (a, b, c)]
    for funcs in product(*[funcs] * 3):
        res = [int(f(k)) for f, k in zip(funcs, (a, b, c))]
        epsilon_new = elems - np.prod(res)
        if np.abs(epsilon_new) < epsilon and all(r >= min_vert for r in res):
            if not strict_max or epsilon_new >= 0:
                result = res
                epsilon = np.abs(epsilon_new)
    result = np.array(result).astype(int)
    if return_elems:
        result = result, np.prod(result)
    return result


def mesh_cuboid(cuboid, target_elems):
    """
    Splits Cuboid up into small Cuboid cells

    Parameters
    ----------
    cuboid: magpylib.Cuboid object
        input object to be discretized

    target_elems: target number of divisions

    Returns
    -------
    discretization: magpylib.Collection
        Collection of Cuboid cells
    """

    # load cuboid properties
    pos = cuboid.position
    rot = cuboid.orientation
    dim = cuboid.dimension
    mag = cuboid.magnetization

    nnn, elems = infer_elems_from_dim(dim, target_elems, return_elems=True)
    print(
        f"Meshing Cuboid with {nnn[0]}x{nnn[1]}x{nnn[2]}={elems} elements (target={target_elems})"
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

    return magpy.Collection(cells)


def mesh_cylinder(cylinder, target_elems):
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

    (nphi, nr, nh), elems = infer_elems_from_dim(dim, target_elems, return_elems=True)
    print(
        f"Meshing CylinderSegement with {nphi}x{nr}x{nh}={elems} elements (target={target_elems})"
    )
    r = np.linspace(r1, r2, nr + 1)
    dh = h / nh
    cyl_segs = []
    for r_ind in range(nr):
        nphi_r = max(1, int(r[r_ind + 1] / ((r1 + r2) / 2) * nphi))
        phi = np.linspace(phi1, phi2, nphi_r + 1)
        for h_ind in range(nh):
            pos_h = dh * h_ind - h / 2 + dh / 2
            if r[r_ind] == 0 and phi2 - phi1 == 360:
                cylinder_class = magpy.magnet.Cylinder
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


def demag_tensor(src_list, store=False, load=False):
    """
    Compute the demagnetization tensor T based on point matching (see Chadbec 2006)
    for n sources in the input collection.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    store: `False` or filename (str)
        Store T after computation as filename.npy.

    load: `False` or filename (str)
        Try to load T from filename.npy.

    Returns
    -------
    Demagnetization tensor: ndarray, shape (3,n,n,3)

    TODO: allow multi-point matching
    TODO: allow current sources
    TODO: allow external stray fields
    TODO: status bar when n>1000
    TODO: Speed up with direct interface for field computation
    TODO: Use newell formulas for cube-cube interactions
    """
    n = len(src_list)

    # load pre-computed tensor
    if isinstance(load, str):
        try:
            T = np.load(load + '.npy')
            print(' - load pre-computed demagnetization tensor')
            if n != T.shape[1]:
                raise ValueError('Loaded demag tensor is not of same shape as input collection')
            return T
        except FileNotFoundError:
            print(' - file not found')

    print(" - computing demagnetization tensor")

    # compute cell positions
    pos = np.empty((n,3))
    for i,src in enumerate(src_list):
        if isinstance(src, magpy.magnet.CylinderSegment):
            pos[i] = src.barycenter
        else:
            pos[i] = src.position

    # split up magnetizations
    coll3 = magpy.Collection()
    for unit_mag in [(1,0,0), (0,1,0), (0,0,1)]:
        for src in src_list:
            src.magnetization=src.orientation.inv().apply(unit_mag)   # ROTATION CHECK
            coll3.add(src.copy())

    # point matching field and demag tensor
    Hpoint = magpy.getH(coll3.sources, pos) # shape (3n cells, n pos, 3 xyz)
    T = Hpoint.reshape(3, n, n, 3) # shape (3 unit mag, n cells, n pos, 3 xyz)

    # store tensor
    if isinstance(store, str):
        np.save(store + '.npy', T)

    return T


def invert(matrix, solver):
    """
    Matrix inversion

    Parameters
    ----------
    matrix: np.array, shape (n,n)
        Input matrix to be inverted.

    solver: str
        Solver to be used. Must be one of (np.linalg.inv, ).

    Returns
    -------
    matrix_inverse: ndarray, shape (n,n)


    TODO implement and test different solver packages
    TODO check input matrix and auto-select correct solver (direct, iterative, ...)
    """
    print(' - solving with '+ solver)

    if solver == 'np.linalg.inv':
        return np.linalg.inv(matrix)

    raise ValueError('Bad solver input.')


def apply_demag(
    collection,
    xi,
    solver='np.linalg.inv',
    demag_store=False,
    demag_load=False,
    ):
    '''
    Computes the interaction between all collection magnets and fixes their magnetization.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    xi: array_like, shape (n,)
        Vector of n magnetic susceptibilities of the cells.

    solver: str, default='np.linalg.inv'
        Solver to be used. Must be one of (np.linalg.inv, ).

    demag_store: `False` or filename (str)
        Store demagnetization tensor T after computation as filename.npy.

    demag_load: `False` or filename (str)
        Try to load demagnetization tensor T from filename.npy.

    Returns
    -------
    None
    '''
    n = len(collection.sources_all)

    print(f'Starting demag computation with {n} cells.')

    # set up mr
    mag = [src.orientation.apply(src.magnetization) for src in collection.sources_all]    # ROTATION CHECK
    mag = np.reshape(mag,(3*n,1), order='F')   # shape ii = x1, ... xn, y1, ... yn, z1, ... zn

    # set up S
    xi = np.array(xi)
    if len(xi) != n:
        raise ValueError('Apply_demag input collection and xi must have same length.')
    S = np.diag(np.tile(xi, 3)) # shape ii, jj

    # set up T
    T = demag_tensor(
        collection.sources_all,
        store=demag_store,
        load=demag_load,
    ) # shape (3 mag unit, n cells, n positions, 3 Bxyz)
    #T = T.swapaxes(0, 3)
    T = T*(4*np.pi/10)
    T = T.swapaxes(2, 3)
    T = np.reshape(T, (3*n,3*n)).T # shape ii, jj

    # set up and invert Q
    Q = np.eye(3*n) - np.matmul(S, T)
    Q_inv = invert(matrix=Q, solver=solver)

    # determine new magnetization vectors
    mag_new = np.matmul(Q_inv, mag)
    mag_new = np.reshape(mag_new, (n,3), order='F')
    #mag_new *= .4*np.pi

    for s,mag in zip(collection.sources_all, mag_new):
        s.magnetization = s.orientation.inv().apply(mag)   # ROTATION CHECK

    print('Demag computation completed.')
