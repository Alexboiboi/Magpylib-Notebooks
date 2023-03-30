"""demag_functions"""
# +
# pylint: disable=invalid-name, redefined-outer-name, protected-access
import sys
import threading
from collections import Counter
from contextlib import contextmanager
import time

import magpylib as magpy
import numpy as np
from loguru import logger
from meshing_functions import mesh_Cuboid
from scipy.spatial.transform import Rotation as R

from magpylib.magnet import Cuboid
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet, BaseCurrent

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


class ElapsedTimeThread(threading.Thread):
    """ "Stoppable thread that logs the time elapsed"""

    def __init__(self, msg=None, min_log_time=1):
        super(ElapsedTimeThread, self).__init__()
        self._stop_event = threading.Event()
        self.thread_start = time.time()
        self.msg = msg
        self.min_log_time = min_log_time
        self._msg_displayed = False

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def getStart(self):
        return self.thread_start

    def run(self):
        self.thread_start = time.time()
        while not self.stopped():
            if (
                self.msg is not None
                and time.time() - self.thread_start > self.min_log_time
                and not self._msg_displayed
            ):
                logger.opt(colors=True).info(f"Start {self.msg}")
                self._msg_displayed = True
            # include a delay here so the thread doesn't uselessly thrash the CPU
            time.sleep(self.min_log_time / 5)


@contextmanager
def loguru_catchtime(msg, min_log_time=1) -> float:
    """ "Measure and log time with loguru as context manager."""
    start = time.perf_counter()
    end = None
    threadTimer = ElapsedTimeThread(msg=msg, min_log_time=min_log_time)
    threadTimer.start()
    try:
        yield
        end = time.perf_counter() - start
    finally:
        threadTimer.stop()
        threadTimer.join()
        if end is None:
            logger.opt(colors=True).exception(f"{msg} failed")

    if end > min_log_time:
        logger.opt(colors=True).success(
            f"{msg} done" f"<green> 🕑 {round(end, 3)}sec</green>"
        )


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
        mask_inds, getH_params, pos0, rot0 = filter_distance(
            src_list, max_dist, return_params=False, return_base_geo=True
        )
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
        with loguru_catchtime(f"getH with unit_mag={unit_mag}"):
            if pairs_matching or max_dist != 0:
                magnetization = np.repeat(mag_all, len(src_list), axis=0)[mask_inds]
                H_unique = magpy.getH(
                    "Cuboid", magnetization=magnetization, **getH_params
                )
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

    # shape (3_unit_mag, n_cells, n_pos, 3_xyz)
    T = np.array(H_point).reshape((3, n, n, 3))

    # store tensor
    if isinstance(store, str):
        fn = store + ".npy"
        logger.success(f"Saved demagnetization tensor to {fn}")
        np.save(fn, T)

    return T


def filter_distance(src_list, max_dist, return_params=False, return_base_geo=False):
    """filter indices by distance parameter"""
    with loguru_catchtime("Distance filter"):
        all_cuboids = all(isinstance(src, Cuboid) for src in src_list)
        if not all_cuboids:
            raise ValueError(
                "filter_distance only implemented if all sources are Cuboids"
            )
        pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
        rotQ0 = [src.orientation.as_quat() for src in src_list]
        rot0 = R.from_quat(rotQ0)
        dim0 = [src.dimension for src in src_list]

        pos2 = np.tile(pos0, (len(pos0), 1)) - np.repeat(pos0, len(pos0), axis=0)
        dist2 = np.linalg.norm(pos2, axis=1)
        dim2 = np.tile(dim0, (len(dim0), 1)), np.repeat(dim0, len(dim0), axis=0)
        maxdim2 = np.concatenate(dim2, axis=1).max(axis=1)
        mask = (dist2 / maxdim2) < max_dist
        if return_params:
            params = dict(
                observers=np.tile(pos0, (len(src_list), 1))[mask],
                position=np.repeat(pos0, len(src_list), axis=0)[mask],
                orientation=R.from_quat(np.repeat(rotQ0, len(src_list), axis=0))[mask],
                dimension=np.repeat(dim0, len(src_list), axis=0)[mask],
            )
        dsf = sum(mask) / len(mask) * 100
    log_msg = f"Interaction pairs left after distance factor filtering: <blue>{dsf:.2f}%</blue>"
    if dsf == 0:
        logger.opt(colors=True).warning(log_msg)
    else:
        logger.opt(colors=True).success(log_msg)
    out = [mask]
    if return_params:
        out.append(params)
    if return_base_geo:
        out.extend([pos0, rot0])
    if len(out) == 1:
        return out[0]
    return tuple(out)


def match_pairs(src_list):
    """match all pairs of sources from `src_list`"""
    with loguru_catchtime("Pairs matching"):
        all_cuboids = all(isinstance(src, Cuboid) for src in src_list)
        if not all_cuboids:
            raise ValueError(
                "Pairs matching only implemented if all sources are Cuboids"
            )
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
            prop = (np.concatenate([pos2, rotQ2a, rotQ2b, dim2], axis=1) + 1e-9).round(
                8
            )
            logger.info("find unique indices")
            _, unique_inds, unique_inv_inds = np.unique(
                prop, return_index=True, return_inverse=True, axis=0
            )
            perc = len(unique_inds) / len(unique_inv_inds) * 100
            logger.opt(colors=True).info(
                f"Interaction pairs left after pair matching filtering: <blue>{perc:.2f}%</blue>"
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
    with loguru_catchtime(f"Matrix inversion with <blue>{solver}</blue>"):
        if solver == "np.linalg.inv":
            res = np.linalg.inv(quat)
            return res

    raise ValueError("Bad solver input.")


def find_sources_to_refine(src_list, mag_diff_thresh=500, max_dist=1.5):
    """Return a set of sources from `src_list` which meet following criteria for refinement:"
    - relative-to-dimension pair of sources < `max_dist`
    - norm(abs(mag1 -mag2)) > `mag_diff_thresh`
    """
    len_src = len(src_list)
    src_tile = np.tile(src_list, len_src)
    src_repeat = np.repeat(src_list, len_src)

    dist_mask, pos0, rot0 = filter_distance(
        src_list, max_dist=max_dist, return_base_geo=True
    )
    mag0 = [src.magnetization for src in src_list]
    magr0 = rot0.apply(mag0)
    mag2 = np.tile(magr0, (len_src, 1)) - np.repeat(magr0, len_src, axis=0)
    mag_mask = np.linalg.norm(np.abs(mag2), axis=1) > mag_diff_thresh
    full_mask = dist_mask & mag_mask
    srcs_to_refine = set(src_tile[full_mask]).intersection(src_repeat[full_mask])
    return srcs_to_refine


def find_sources_to_refine2(src_list, mag_diff_thresh=10, max_dist=1):
    """Return a set of sources from `src_list` which meet following criteria for refinement:"
    - relative-to-dimension pair of sources < `max_dist`
    - norm(abs(mag1 -mag2)) > `mag_diff_thresh`
    """
    len_src = len(src_list)
    src_tile = np.tile(src_list, len_src)
    src_repeat = np.repeat(src_list, len_src)

    dist_mask, pos0, rot0 = filter_distance(
        src_list, max_dist=max_dist, return_base_geo=True
    )
    mag0 = [src.magnetization for src in src_list]
    magr0 = rot0.apply(mag0)
    mag2 = np.tile(magr0, (len_src, 1)) - np.repeat(magr0, len_src, axis=0)
    mag2 = mag2[dist_mask]
    src_tile = src_tile[dist_mask]
    src_repeat = src_repeat[dist_mask]
    mag_norm2 = np.linalg.norm(np.abs(mag2), axis=1)
    # create index mask for the worst `mag_diff_thresh` percents, AFTER max_dist filtering
    mag_mask = np.argsort(mag_norm2) >= min(
        len(mag_norm2) - 1, (100 - mag_diff_thresh) / 100 * len(mag_norm2) - 1
    )
    srcs_to_refine = set(src_tile[mag_mask]).intersection(set(src_repeat[mag_mask]))
    return srcs_to_refine


def apply_demag_with_refinement(
    collection,
    inplace=False,
    init_refine_factor=8,
    refine_factor=2,
    max_dist=2,
    mag_diff_thresh=500,
    max_passes=10,
    max_elems=None,
):
    """apply demag iteratively and recursively with refinement options"""
    if not inplace:
        coll = collection.copy()
    else:
        coll = collection

    # make initial refinement
    if init_refine_factor > 1:
        refine(*coll.sources_all, factor=init_refine_factor)

    # initialize
    pass_num = 0
    srcs_to_refine = True
    # main loop

    while pass_num < max_passes and srcs_to_refine:
        pass_num += 1
        logger.opt(colors=True).info(
            f"Adaptive pass <blue>{pass_num} (max={max_passes})</blue>"
        )
        src_list = coll.sources_all
        # store magnetizations before applying demag, to start next iteration with
        # correct magnetizations
        mags_before_demag = np.array([src._magnetization for src in src_list])
        apply_demag(coll, inplace=True)
        # only apply refinement if necessary, last step loop must be demag
        if pass_num != max_passes and srcs_to_refine:
            srcs_to_refine = find_sources_to_refine(
                src_list, mag_diff_thresh=mag_diff_thresh, max_dist=max_dist
            )
            new_extra_src = refine_factor * len(srcs_to_refine)
            if len(srcs_to_refine) == 0:
                logger.opt(colors=True).success(
                    "No Sources to be refined, <red>stopping adaptive passes</red>"
                )
            elif max_elems is not None and len(src_list) + new_extra_src > max_elems:
                logger.opt(colors=True).info(
                    f"Refinement stopped with {len(src_list)} cells ({max_elems=})."
                    f" Further adaptive pass would result in {len(src_list) + new_extra_src} cells."
                )
                break
            else:
                a, b = len(srcs_to_refine), len(src_list)
                logger.opt(colors=True).success(
                    f" Sources refined : <blue>{len(srcs_to_refine)}/{len(src_list)} ({a/b*100:.2f}%)</blue>"
                )
                for src, mag in zip(src_list, mags_before_demag):
                    src._magnetization = mag
                refine(*srcs_to_refine, factor=refine_factor)
    if not inplace:
        return coll


def refine(*sources, factor, mode="cuboids"):
    """refine sources and replace them inplace with a meshed collection"""
    if mode != "cuboids":
        raise ValueError("Refinement only supports 'cuboids' mode")
    for src in sources:
        parent = src.parent
        src.parent = None
        meshed_coll = mesh_Cuboid(src, factor)
        for child in meshed_coll:
            child.xi = src.xi
        parent.add(meshed_coll)


def apply_demag(
    collection,
    xi=None,
    solver="np.linalg.inv",
    demag_store=False,
    demag_load=False,
    inplace=False,
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
    if not inplace:
        collection = collection.copy()
    if style is not None:
        collection.style = style
    srcs = collection.sources_all
    magnets_list = [src for src in srcs if isinstance(src, BaseMagnet)]
    currents_list = [src for src in srcs if isinstance(src, BaseCurrent)]
    others_list = [
        src
        for src in srcs
        if not isinstance(src, (BaseMagnet, BaseCurrent, magpy.Sensor))
    ]
    if others_list:
        raise TypeError(
            "Only Magnet and Current sources supported. "
            "Incompatible objects found: "
            f"{Counter(s.__class__.__name__ for s in others_list)}"
        )
    n = len(magnets_list)
    counts = Counter(s.__class__.__name__ for s in magnets_list)
    inplace_str = f"""{" (inplace) " if inplace else " "}"""
    demag_msg = (
        f"Demagnetization computation{inplace_str}of <blue>{collection}</blue>"
        f" with {n} cells - {counts}"
    )
    with loguru_catchtime(demag_msg):
        # set up mr
        mag = [
            src.orientation.apply(src.magnetization) for src in magnets_list
        ]  # ROTATION CHECK
        mag = np.reshape(
            mag, (3 * n, 1), order="F"
        )  # shape ii = x1, ... xn, y1, ... yn, z1, ... zn

        # set up S
        if xi is None:
            xi = get_xi(*magnets_list)
        xi = np.array(xi)
        if len(xi) != n:
            raise ValueError(
                "Apply_demag input collection and xi must have same length."
            )
        S = np.diag(np.tile(xi, 3))  # shape ii, jj

        # set up T (3 mag unit, n cells, n positions, 3 Bxyz)
        with loguru_catchtime("Demagnetization tensor calculation"):
            T = demag_tensor(
                magnets_list,
                store=demag_store,
                load=demag_load,
                split=split,
                pairs_matching=pairs_matching,
                max_dist=max_dist,
            )

        T = T * (4 * np.pi / 10)
        T = T.swapaxes(2, 3)
        T = np.reshape(T, (3 * n, 3 * n)).T  # shape ii, jj

        # Incorporate the magnetic field contributions from current sources
        pos = np.array([src.position for src in magnets_list])
        mag_currents = magpy.getH(currents_list, pos)
        mag_currents = np.reshape(mag_currents, (3 * n, 1), order="F")

        # set up Q
        Q = np.eye(3 * n) - np.matmul(S, T)

        with loguru_catchtime("Solving of linear system"):
            mag_new = np.linalg.solve(Q, mag + np.matmul(S, mag_currents))

        mag_new = np.reshape(mag_new, (n, 3), order="F")
        # mag_new *= .4*np.pi

        for s, mag in zip(collection.sources_all, mag_new):
            s.magnetization = s.orientation.inv().apply(mag)  # ROTATION CHECK

    if not inplace:
        return collection
