"""demag_functions"""
# +
# pylint: disable=invalid-name, redefined-outer-name, protected-access

import sys
from time import perf_counter
from collections import Counter

from loguru import logger
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

config = {
    "handlers": [
        dict(
            sink=sys.stdout,
            colorize=True,
            format=(
                "<magenta>{time:YYYY-MM-DD at HH:mm:ss}</magenta>"
                " | <level>{level:^8}</level>"
                " | <cyan>{function}</cyan>"
                " | <yellow>{extra}</yellow> {level.icon:<2} {message}"
            ),
        ),
    ],
}
logger.configure(**config)

# -


def get_xi(*sources, xi=None):
    """Return a list of length (len(sources)) with xi values
    Priority is given at the source level, hovever if value is not found, it is searched up the
    the parent tree, if available. Raises an error if no value is found when reached the top
    level of the tree"""
    xis = []
    for src in sources:
        xi = getattr(src, "xi", None)
        if xi is None:
            if src.parent is None:
                raise ValueError(
                    "No susceptibility `xi` defined in any parent collection"
                )
            xis.extend(get_xi(src.parent))
        else:
            xis.append(xi)
    return xis


def demag_tensor(
    src_list, store=False, load=False, pairs_matching=False, split=False, max_dist=0
):
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

    pairs_matching: bool
        If True, equivalent pair of interactions are identified and unique pairs are
        calculated only once and copied to duplicates.

    split: int
        Number of times the sources list is splitted before getH calculation ind demag
        tensor calculation

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
        logger.info("load pre-computed demagnetization tensor")
        if n != T.shape[1]:
            raise ValueError(
                "Loaded demag tensor is not of same shape as input collection"
            )
        return T

    if pairs_matching and split != 1:
        raise ValueError("Pairs matching does not support splitting")
    elif max_dist != 0:
        getH_params, mask_inds, pos0, rot0 = filter_distance(src_list, max_dist)
    elif pairs_matching:
        getH_params, mask_inds, unique_inv_inds, pos0, rot0 = match_pairs(src_list)
    else:
        pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
        rotQ0 = [src.orientation.as_quat() for src in src_list]
        rot0 = R.from_quat(rotQ0)

    H_point = []
    for unit_mag in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        mag_all = rot0.inv().apply(unit_mag)
        # point matching field and demag tensor
        getH_time = perf_counter()
        if pairs_matching or max_dist != 0:
            magnetization = np.repeat(mag_all, len(src_list), axis=0)[mask_inds]
            H_unique = magpy.getH("Cuboid", magnetization=magnetization, **getH_params)
            if max_dist != 0:
                H_temp = np.zeros((len(src_list) ** 2, 3))
                H_temp[mask_inds] = H_unique
                H_unit_mag = H_temp
            else:
                H_unit_mag = H_unique[unique_inv_inds]
        else:
            for src, mag in zip(src_list, mag_all):
                src.magnetization = mag
            if split > 1:
                src_list_split = np.array_split(src_list, split)
                with logger.contextualize(
                    task="Splitting field calculation", split=split
                ):
                    H_unit_mag = []
                    for split_ind, src_list_subset in enumerate(src_list_split):
                        logger.info(
                            f"Sources subset {split_ind+1}/{len(src_list_split)}"
                        )
                        if src_list_subset.size > 0:
                            H_unit_mag.append(
                                magpy.getH(src_list_subset.tolist(), pos0)
                            )
                    H_unit_mag = np.concatenate(H_unit_mag, axis=0)
            else:
                H_unit_mag = magpy.getH(src_list, pos0)
        H_point.append(H_unit_mag)  # shape (n_cells, n_pos, 3_xyz)
        logger.opt(colors=True).success(
            f"getH with unit_mag={unit_mag} done"
            f"<green> 🕑 {round(perf_counter()-getH_time, 3)}sec</green>"
        )

    # shape (3_unit_mag, n_cells, n_pos, 3_xyz)
    T = np.array(H_point).reshape((3, n, n, 3))

    # store tensor
    if isinstance(store, str):
        fn = store + ".npy"
        logger.success(f"Saved demagnetization tensor to {fn}")
        np.save(fn, T)

    return T


def filter_distance(src_list, max_dist):
    """filter indices by distance parameter"""
    filter_distance_time = perf_counter()
    all_cuboids = all(src._object_type == "Cuboid" for src in src_list)
    if not all_cuboids:
        raise ValueError("filter_distance only implemented if all sources are Cuboids")
    pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
    rotQ0 = [src.orientation.as_quat() for src in src_list]
    rot0 = R.from_quat(rotQ0)
    dim0 = [src.dimension for src in src_list]

    pos2 = np.tile(pos0, (len(pos0), 1)) - np.repeat(pos0, len(pos0), axis=0)
    dist2 = np.linalg.norm(pos2, axis=1)
    dim2 = np.tile(dim0, (len(dim0), 1)), np.repeat(dim0, len(dim0), axis=0)
    maxdim2 = np.concatenate(dim2, axis=1).max(axis=1)
    mask = (dist2 / maxdim2) < max_dist
    params = dict(
        observers=np.tile(pos0, (len(src_list), 1))[mask],
        position=np.repeat(pos0, len(src_list), axis=0)[mask],
        orientation=R.from_quat(np.repeat(rotQ0, len(src_list), axis=0))[mask],
        dimension=np.repeat(dim0, len(src_list), axis=0)[mask],
    )
    dsf = sum(~mask) / len(mask) * 100
    log_msg = (
        f"Distance factor savings: <blue>{dsf:.2f}%</blue>"
        f"<green> 🕑 {round(perf_counter()-filter_distance_time, 3)}sec</green>"
    )
    if dsf == 0:
        logger.opt(colors=True).warning(log_msg)
    else:
        logger.opt(colors=True).success(log_msg)
    return params, mask, pos0, rot0


def match_pairs(src_list):
    """match all pairs of sources from `src_list`"""
    match_pairs_time = perf_counter()
    all_cuboids = all(src._object_type == "Cuboid" for src in src_list)
    if not all_cuboids:
        raise ValueError("Pairs matching only implemented if all sources are Cuboids")
    pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
    rotQ0 = [src.orientation.as_quat() for src in src_list]
    rot0 = R.from_quat(rotQ0)
    dim0 = [src.dimension for src in src_list]
    len_src = len(src_list)
    num_of_pairs = len_src**2
    with logger.contextualize(task="Match interactions pairs"):
        logger.info("position")
        pos2 = np.tile(pos0, (len_src, 1)) - np.repeat(pos0, len_src, axis=0)
        logger.info("orientation")
        rotQ2a = np.tile(rotQ0, (len_src, 1)).reshape((num_of_pairs, -1))
        rotQ2b = np.repeat(rotQ0, len_src, axis=0).reshape((num_of_pairs, -1))
        logger.info("dimension")
        dim2 = np.tile(dim0, (len_src, 1)) - np.repeat(dim0, len_src, axis=0)
        logger.info("concatenate properties")
        prop = (np.concatenate([pos2, rotQ2a, rotQ2b, dim2], axis=1) + 1e-9).round(8)
        logger.info("find unique indices")
        _, unique_inds, unique_inv_inds = np.unique(
            prop, return_index=True, return_inverse=True, axis=0
        )
        sav_perc = 100 - len(unique_inds) / len(unique_inv_inds) * 100
        logger.opt(colors=True).success(
            f"Pair matching savings: <blue>{sav_perc:.2f}%</blue>"
            f"<green> 🕑 {round(perf_counter()-match_pairs_time, 3)}sec</green>"
        )

    params = dict(
        observers=np.tile(pos0, (len(src_list), 1))[unique_inds],
        position=np.repeat(pos0, len(src_list), axis=0)[unique_inds],
        orientation=R.from_quat(rotQ2b)[unique_inds],
        dimension=np.repeat(dim0, len(src_list), axis=0)[unique_inds],
    )
    return params, unique_inds, unique_inv_inds, pos0, rot0


def invert(quat, solver):
    """
    quat inversion

    Parameters
    ----------
    quat: np.array, shape (n,n)
        Input quat to be inverted.

    solver: str
        Solver to be used. Must be one of (np.linalg.inv, ).

    Returns
    -------
    quat_inverse: ndarray, shape (n,n)


    TODO implement and test different solver packages
    TODO check input quat and auto-select correct solver (direct, iterative, ...)
    """
    quat_inv_start_time = perf_counter()
    logger.info("Start solving with " + solver)

    if solver == "np.linalg.inv":
        res = np.linalg.inv(quat)
        logger.opt(colors=True).success(
            f"quat inversion done"
            f"<green> 🕑 {round(perf_counter()- quat_inv_start_time, 3)}sec</green>"
        )
        return res

    raise ValueError("Bad solver input.")


def apply_demag(
    collection,
    xi=None,
    solver="np.linalg.inv",
    demag_store=False,
    demag_load=False,
    inplace=True,
    pairs_matching=False,
    max_dist=0,
    split=1,
    style=None,
):
    """
    Computes the interaction between all collection magnets and fixes their magnetization.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    xi: array_like, shape (n,)
        Vector of n magnetic susceptibilities of the cells. If not defined, values are searched at
        object level or parent level if needed.

    solver: str, default='np.linalg.inv'
        Solver to be used. Must be one of (np.linalg.inv, ).

    demag_store: `False` or filename (str)
        Store demagnetization tensor T after computation as filename.npy.

    demag_load: `False` or filename (str)
        Try to load demagnetization tensor T from filename.npy.

    inplace: bool
        If True, applies demagnetization on a copy of the input collection and returns the
        demagnetized collection

    pairs_matching: bool
        If True, equivalent pair of interactions are identified and unique pairs are
        calculated only once and copied to duplicates. This parameter is not compatible with
        `max_dist` or `split` and applies only cuboid cells.

    max_dist: float
        Posivive number representing the max_dimension to distance ratio for each pair of
        interacting cells. This filters out far interactions. If `max_dist=0`, all interactions are
        calculated. This parameter is not compatible with `pairs_matching` or `split` and applies
        only cuboid cells.

    split: int
        Number of times the sources list is splitted before getH calculation ind demag
        tensor calculation. This parameter is not compatible with `pairs_matching` or `max_dist`.

    style: dict
        Set collection style. If `inplace=False` only affects the copied collection

    Returns
    -------
    None
    """
    demag_start_time = perf_counter()
    if not inplace:
        collection = collection.copy()
    if style is not None:
        collection.style = style

    src_list = collection.sources_all
    n = len(src_list)
    counts = Counter(s._object_type for s in src_list)
    logger.opt(colors=True).info(
        f"Start demagnetization computation of <blue>{collection}</blue> with {n} cells - {counts}"
        f""" {"(inplace)" if inplace else ""}"""
    )

    # set up mr
    mag = [
        src.orientation.apply(src.magnetization) for src in src_list
    ]  # ROTATION CHECK
    mag = np.reshape(
        mag, (3 * n, 1), order="F"
    )  # shape ii = x1, ... xn, y1, ... yn, z1, ... zn

    # set up S
    if xi is None:
        xi = get_xi(*src_list)
    xi = np.array(xi)
    if len(xi) != n:
        raise ValueError("Apply_demag input collection and xi must have same length.")
    S = np.diag(np.tile(xi, 3))  # shape ii, jj

    # set up T (3 mag unit, n cells, n positions, 3 Bxyz)
    demag_tensor_start_time = perf_counter()
    logger.info("Start demagnetization tensor calculation")
    T = demag_tensor(
        src_list,
        store=demag_store,
        load=demag_load,
        split=split,
        pairs_matching=pairs_matching,
        max_dist=max_dist,
    )
    logger.opt(colors=True).success(
        f"Demagnetization tensor calculation done"
        f"<green> 🕑 {round(perf_counter()- demag_tensor_start_time, 3)}sec</green>"
    )
    # T = T.swapaxes(0, 3)
    T = T * (4 * np.pi / 10)
    T = T.swapaxes(2, 3)
    T = np.reshape(T, (3 * n, 3 * n)).T  # shape ii, jj

    # set up and invert Q
    Q = np.eye(3 * n) - np.matmul(S, T)
    Q_inv = invert(quat=Q, solver=solver)

    # determine new magnetization vectors
    mag_new = np.matmul(Q_inv, mag)
    mag_new = np.reshape(mag_new, (n, 3), order="F")
    # mag_new *= .4*np.pi

    for s, mag in zip(collection.sources_all, mag_new):
        s.magnetization = s.orientation.inv().apply(mag)  # ROTATION CHECK

    logger.opt(colors=True).success(
        f"Demagnetization computation done"
        f"<green> 🕑 {round(perf_counter()- demag_start_time, 3)}sec</green>"
    )
    if not inplace:
        return collection
