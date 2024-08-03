from pathlib import Path
from typing import List

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cmapvals(vals, cmap="viridis", vmin=None, vmax=None):
    """Maps a list of values to corresponding RGB values in a matplotlib colormap."""
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
