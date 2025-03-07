import numpy as np
from tqdm import tqdm


def ransac(*data, fit_func=None, loss_func=None, cost_func=None, samp_min=10, inlier_min=10, inlier_thres=0.1, max_iter=1000, seed=None):
    """
    Standard RANSAC algorithm for arbitrary functions.

    Args:
        fit_func (data -> model)
        loss_func ((model, data) -> array): vectorized loss for individual data points
        cost_func ((model, data) -> scalar): total cost to try to minimize
    
    Returns:
        model, inlier_idxs
    """
    rng = np.random.default_rng(seed)
    best_model = None
    best_inlier_idxs = []
    best_inliers = []
    best_error = float("inf")

    for _ in tqdm(range(max_iter), desc="Running RANSAC"):
        sample_indices = rng.choice(len(data[0]), samp_min, replace=False)
        sample = [singledata[sample_indices] for singledata in data]

        model = fit_func(sample)

        errors = loss_func(model, data)

        inlier_idxs = np.where(errors < inlier_thres)[0]
        n_inliers = len(inlier_idxs)
        inliers = [singledata[inlier_idxs] for singledata in data]

        total_error = cost_func(model, inliers)

        if n_inliers >= inlier_min:
            if n_inliers > len(best_inlier_idxs) or (n_inliers == len(best_inlier_idxs) and total_error < best_error):
                best_model = model
                best_inliers = inliers
                best_inlier_idxs = inlier_idxs
                best_error = total_error
    # TODO add condition to relax constraints if this is reached
    if best_model is None:
        raise ValueError("No valid model found after RANSAC")
    return best_model, best_inlier_idxs
