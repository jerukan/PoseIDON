import json
import math
from pathlib import Path
from typing import List, Dict, Union

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def name_idx_from_paths(name: str, paths: List[Path]):
    names = [path.stem for path in paths]
    if name not in names:
        return -1
    return names.index(name)


def invert_idxs(idxs, n):
    """
    Invert numerical index.
    """
    allidxs = np.arange(n)
    mask = np.ones(n, dtype=bool)
    mask[idxs] = False
    return allidxs[mask]


def add_to_json(data: Dict, path: Union[Path, str]):
    """
    Adds data or modifies existing keys to a JSON file.
    """
    with open(path, "rt") as f:
        jsondata = yaml.safe_load(f)
    jsondata = {
        **jsondata,
        # put second to overwrite existing keys
        **data,
    }
    with open(path, "wt") as f:
        json.dump(jsondata, f, indent=4)


def combine_path_tail(parent, tail, taillen: int):
    """
    Appends part of the tail of a path to a given parent.

    Args:
        taillen (int): number of parts starting from the tail to append from the parent
    """
    parent = Path(parent)
    tail = Path(tail)
    return parent / Path(*tail.parts[-taillen:])


def ext_pattern(extension):
    """
    Because I want to use glob instead of looping through files.

    Example: "jpg" -> "*.[jJ][pP][gG]"
    """
    return "*." + "".join("[%s%s]" % (e.lower(), e.upper()) for e in extension)


def index_array_or_list(arr_or_list, elem) -> int:
    """
    Like list.index() but for numpy arrays too. Will only return the first index.
    """
    if isinstance(arr_or_list, np.ndarray):
        return np.where(arr_or_list == elem)[0][0]
    elif isinstance(arr_or_list, list):
        return arr_or_list.index(elem)
    raise TypeError()


def match_lists(*lists: List[List]) -> List[int]:
    """
    Finds the indices of elements that are common to all lists.

    The returned indices will index elements from each list in the same order.
    """
    reflist = lists[0]
    otherlists = lists[1:]
    matchidxs: List[List] = [[] for _ in lists]
    for i, elem in enumerate(reflist):
        inalllists = True
        for j, otherlist in enumerate(otherlists):
            if elem not in otherlist:
                inalllists = False
                break
        if inalllists:
            matchidxs[0].append(i)
            for j, otherlist in enumerate(otherlists):
                matchidxs[j + 1].append(index_array_or_list(otherlist, elem))
    return matchidxs


def cmapvals(vals, cmap="viridis", vmin=None, vmax=None):
    """
    Maps a list of values to corresponding RGB values in a matplotlib colormap.

    Returns:
        nx3 array of float RGB values
    """
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.min(vals)
    if vmax is None:
        vmax = np.max(vals)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgbvals = np.array(scalarMap.to_rgba(vals))
    rgbvals = rgbvals[:, :3]
    return rgbvals


def denoise_nav_depth(df: pd.DataFrame, thresh=6, iters=1) -> pd.DataFrame:
    """
    Remove outlier noisy depth values that are adjacent to valid values iteratively.

    The assumption of the noise is that the depth noise only jumps to shallower depths
    erroneously.
    """
    valid_nav = df
    for _ in range(iters):
        good = np.diff(valid_nav["depth"]) > -thresh
        good = np.insert(good, 0, True)
        good2 = np.diff(valid_nav["depth"]) < thresh
        good2 = np.insert(good2, -1, True)
        valid_nav = valid_nav[good & good2]
    return valid_nav


def random_unitvec3(n=1):
    """
    Generates uniformly distributed 3D unit vector.

    Args:
        n: number of vectors to generate

    Returns:
        nx3 vector
    """
    # apparently the standard multivariate normal distribution
    # is rotation invariant, so its distributed uniformly
    unnormxyzs = np.random.normal(0.0, 1.0, size=(n, 3))
    xyzs = unnormxyzs / np.linalg.norm(unnormxyzs, axis=1)[..., None]
    return xyzs


def rgb2hex(rgb):
    if len(rgb) == 3:
        return "#{:02x}{:02x}{:02x}".format(*rgb)
    elif len(rgb) == 4:
        return "#{:02x}{:02x}{:02x}{:02x}".format(*rgb)
    raise ValueError("Only RGB or RGBA arrays are supported (3 or 4 elements)")


def haversine(lat1, lat2, lon1, lon2):
    """
    Calculates the haversine distance between two points.
    """
    R = 6378.137  # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(lat1 * math.pi / 180) * np.cos(
        lat2 * math.pi / 180
    ) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(a ** (1 / 2), (1 - a) ** (1 / 2))
    d = R * c
    return d * 1000  # meters
