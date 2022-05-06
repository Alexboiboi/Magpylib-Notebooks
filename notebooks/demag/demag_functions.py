# +
# pylint: disable=invalid-name, redefined-outer-name

from time import perf_counter

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation as R


# -


def demag_tensor(src_list, store=False, load=False, verbose=False):
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

    verbose: bool
        If True, prints out demagnetization process informations

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
        T = np.load(load + ".npy")
        if verbose:
            print(" - load pre-computed demagnetization tensor")
        if n != T.shape[1]:
            raise ValueError(
                "Loaded demag tensor is not of same shape as input collection"
            )
        return T

    if verbose:
        print(" - computing demagnetization tensor")

    if verbose:
        print("   - compute cell positions")

    pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
    rotQ0 = [src.orientation.as_quat() for src in src_list]
    mag0 = [src.magnetization for src in src_list]

    if verbose:
        print("   - point matching field and demag tensor")
    Hpoint = []
    for unit_mag in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        mag_all = R.from_quat(rotQ0).inv().apply(unit_mag)  # ROTATION CHECK
        for src, mag in zip(src_list, mag_all):
            src.magnetization = mag  # _magnetization faster safe ?#
        # point matching field and demag tensor
        H = magpy.getH(src_list, pos0)
        Hpoint.append(H)  # shape (n_cells, n_pos, 3_xyz)

    # shape (3_unit_mag, n_cells, n_pos, 3_xyz)
    T = np.array(Hpoint).reshape((3, n, n, 3))

    # store tensor
    if isinstance(store, str):
        fn = store + ".npy"
        if verbose:
            print(f"Saving demagnetization tensor to {fn}")
        np.save(fn, T)

    return T


def invert(matrix, solver, verbose=False):
    """
    Matrix inversion

    Parameters
    ----------
    matrix: np.array, shape (n,n)
        Input matrix to be inverted.

    solver: str
        Solver to be used. Must be one of (np.linalg.inv, ).

    verbose: bool
        If True, prints out process informations

    Returns
    -------
    matrix_inverse: ndarray, shape (n,n)


    TODO implement and test different solver packages
    TODO check input matrix and auto-select correct solver (direct, iterative, ...)
    """
    if verbose:
        print(" - solving with " + solver)

    if solver == "np.linalg.inv":
        return np.linalg.inv(matrix)

    raise ValueError("Bad solver input.")


def apply_demag(
    collection,
    xi,
    solver="np.linalg.inv",
    demag_store=False,
    demag_load=False,
    inplace=True,
    verbose=False,
):
    """
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

    verbose: bool
        If True, prints out demagnetization process informations

    Returns
    -------
    None
    """
    if not inplace:
        collection = collection.copy()
    n = len(collection.sources_all)

    if verbose:
        print(f"Starting demag computation with {n} cells.")

    # set up mr
    mag = [
        src.orientation.apply(src.magnetization) for src in collection.sources_all
    ]  # ROTATION CHECK
    mag = np.reshape(
        mag, (3 * n, 1), order="F"
    )  # shape ii = x1, ... xn, y1, ... yn, z1, ... zn

    # set up S
    xi = np.array(xi)
    if len(xi) != n:
        raise ValueError("Apply_demag input collection and xi must have same length.")
    S = np.diag(np.tile(xi, 3))  # shape ii, jj

    # set up T
    start_time = perf_counter()
    if verbose:
        print(" - Start demagenization tensor calculation")
    T = demag_tensor(
        collection.sources_all,
        store=demag_store,
        load=demag_load,
        verbose=verbose,
    )  # shape (3 mag unit, n cells, n positions, 3 Bxyz)
    if verbose:
        print(
            f" - Finished demagnetization tensor calculation in {round(perf_counter()- start_time, 3)}sec"
        )
    # T = T.swapaxes(0, 3)
    T = T * (4 * np.pi / 10)
    T = T.swapaxes(2, 3)
    T = np.reshape(T, (3 * n, 3 * n)).T  # shape ii, jj

    # set up and invert Q
    Q = np.eye(3 * n) - np.matmul(S, T)
    Q_inv = invert(matrix=Q, solver=solver, verbose=verbose)

    # determine new magnetization vectors
    mag_new = np.matmul(Q_inv, mag)
    mag_new = np.reshape(mag_new, (n, 3), order="F")
    # mag_new *= .4*np.pi

    for s, mag in zip(collection.sources_all, mag_new):
        s.magnetization = s.orientation.inv().apply(mag)  # ROTATION CHECK

    if verbose:
        print("Demag computation completed.")
    if not inplace:
        return collection
