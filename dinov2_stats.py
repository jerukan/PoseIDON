"""
DINOv2 Reliability Analysis for Underwater 6DoF Pose Estimation
================================================================
Analyses produced:

  A4   – Patch descriptor similarity statistics
  A4b  – Feature distance distributions
  A5a  – PnP+RANSAC inlier ratio (per image)
  A5b  – Hypothesis acceptance rate (dataset-level)
  A6a  – BOP metrics vs burial depth / volume ratio
  A6b  – Feature metrics vs BOP errors (rolling median + quantile band)
  A6c  – BOP metrics by burial depth quartile (violin)
  A7   – Template ranking analysis
  A8a  – SIFT pose availability per dataset
  A8b  – Paired DINOv2 vs SIFT BOP scatter
  A8c  – DINOv2 vs SIFT BOP violin by depth quartile
  A8d  – Burial depth vs BOP metric, DINOv2 vs SIFT overlaid
  A8e  – Normalised BOP histograms, DINOv2 (all) vs SIFT (valid only)

Caching
-------
BOP rendering is slow.  On first run the per-image metric rows are saved to
  <cache_dir>/bop_cache.json
On subsequent runs that file is loaded directly and rendering is skipped.
Pass --overwrite to force re-computation and overwrite the cache.

Usage
-----
    python analyze_dinov2_reliability.py               # use defaults + cache
    python analyze_dinov2_reliability.py --overwrite   # recompute BOP metrics
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from scipy import stats
from tqdm import tqdm
import trimesh
import yaml


# ── colour palette ────────────────────────────────────────────────────────────
C_INLIER  = "#2196F3"
C_OUTLIER = "#F44336"
C_BOTH    = "#78909C"
C_GREEN   = "#4CAF50"
C_ORANGE  = "#FF9800"


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────

def stem(img_path: str) -> str:
    return Path(img_path).stem

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def name_idx(stem_name, path_list):
    for i, p in enumerate(path_list):
        if Path(p).stem == stem_name:
            return i
    return -1

def _save(fig, output_dir, filename):
    out = Path(output_dir) / filename
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved -> {out}")

def _scatter_reg(ax, x, y, color, xlabel, ylabel, title, alpha=0.5, s=18):
    valid = ~(np.isnan(x) | np.isnan(y))
    ax.scatter(x[valid], y[valid], alpha=alpha, s=s, color=color, edgecolors="none")
    if valid.sum() > 2:
        m, b, r_v, p, _ = stats.linregress(x[valid], y[valid])
        xl = np.linspace(np.nanmin(x[valid]), np.nanmax(x[valid]), 100)
        ax.plot(xl, m * xl + b, "black", lw=1.5,
                label=f"r = {r_v:.3f},  p = {p:.3g}")
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)

def _scatter_inout(ax, x, y, is_in, xlabel, ylabel, title, alpha=0.55, s=18):
    valid = ~(np.isnan(x) | np.isnan(y))
    colors = np.where(is_in[valid], C_INLIER, C_OUTLIER)
    ax.scatter(x[valid], y[valid], c=colors, alpha=alpha, s=s, edgecolors="none")
    if valid.sum() > 2:
        m, b, r_v, p, _ = stats.linregress(x[valid], y[valid])
        xl = np.linspace(np.nanmin(x[valid]), np.nanmax(x[valid]), 100)
        ax.plot(xl, m * xl + b, "black", lw=1.4,
                label=f"r = {r_v:.3f},  p = {p:.3g}")
    n_in  = is_in[valid].sum()
    n_out = (~is_in[valid]).sum()
    patches = [
        mpatches.Patch(color=C_INLIER,  label=f"Fit-inlier  (N={n_in})"),
        mpatches.Patch(color=C_OUTLIER, label=f"Fit-outlier (N={n_out})"),
    ]
    ax.legend(handles=patches, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)

def _rolling_stats(x_sorted, y_sorted, window_frac=0.1, n_pts=60):
    x_min, x_max = x_sorted[0], x_sorted[-1]
    half = (x_max - x_min) * window_frac / 2.0
    x_grid = np.linspace(x_min + half, x_max - half, n_pts)
    y_mean, y_q25, y_q75 = [], [], []
    for xc in x_grid:
        mask = (x_sorted >= xc - half) & (x_sorted <= xc + half)
        yw = y_sorted[mask]
        yw = yw[~np.isnan(yw)]
        if len(yw) < 3:
            y_mean.append(np.nan); y_q25.append(np.nan); y_q75.append(np.nan)
        else:
            y_mean.append(np.median(yw))
            y_q25.append(np.percentile(yw, 25))
            y_q75.append(np.percentile(yw, 75))
    return x_grid, np.array(y_mean), np.array(y_q25), np.array(y_q75)


# ─────────────────────────────────────────────────────────────────────────────
# BOP raw metric evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _raw_bop_metrics(R_est, t_est, R_gt, t_gt,
                     depth_img, K, renderer, object_name,
                     vtxs, diameter, img_width, syms):
    from bop_toolkit.bop_toolkit_lib.pose_error import vsd, mssd, mspd
    t_est = np.asarray(t_est).reshape(3, 1)
    t_gt  = np.asarray(t_gt ).reshape(3, 1)
    taus  = [0.05 * i for i in range(1, 11)]
    try:
        vsd_vals = vsd(R_est, t_est, R_gt, t_gt, depth_img, K, 0.02, taus, True,
                       diameter, renderer, object_name, cost_type="step")
        mssd_val = mssd(R_est, t_est, R_gt, t_gt, vtxs, syms)
        mspd_val = mspd(R_est, t_est, R_gt, t_gt, K, vtxs, syms)
        return float(np.mean(vsd_vals)), float(mssd_val), float(mspd_val)
    except Exception as exc:
        print(f"    [warn] BOP eval failed: {exc}")
        return (np.nan, np.nan, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / "bop_cache.json"

def _save_cache(rows: list, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir)
    # Convert any non-serialisable floats (nan/inf) to None for JSON
    def _clean(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    clean_rows = [{k: _clean(v) for k, v in row.items()} for row in rows]
    with open(path, "w") as f:
        json.dump(clean_rows, f, indent=2)
    print(f"  Cache saved -> {path}  ({len(rows)} rows)")

def _load_cache(cache_dir: Path) -> list | None:
    path = _cache_path(cache_dir)
    if not path.exists():
        return None
    with open(path) as f:
        rows = json.load(f)
    # Restore None -> nan
    def _restore(v):
        return np.nan if v is None else v
    restored = [{k: _restore(v) for k, v in row.items()} for row in rows]
    print(f"  Cache loaded <- {path}  ({len(restored)} rows)")
    return restored


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def discover_datasets(data_root, results_root, model3d_root, need_bop: bool = True):
    """
    Load per-dataset metadata.

    Parameters
    ----------
    need_bop : bool
        When True (default) load everything needed for BOP rendering:
        depth images, RGB images (for resolution detection), vispy renderers,
        mesh vertices / diameter, and symmetry transforms.
        When False (cache hit) these are skipped; the returned dicts will
        have those keys set to None so callers can detect the omission.
        The fast path still loads info.json, gt_obj2cam.json, camera.json,
        estimated-poses.json, and fit-inliers.json -- everything needed by
        analyses A4-A5 and A7.
    """
    from burybarrel.image import imgs_from_dir

    data_root    = Path(data_root)
    results_root = Path(results_root)
    model3d_root = Path(model3d_root)

    model_info_all: dict = load_json(model3d_root / "model_info.json")
    resolution2renderer: dict = {}
    if need_bop:
        from bop_toolkit.bop_toolkit_lib.misc import get_symmetry_transformations
        from bop_toolkit.bop_toolkit_lib.renderer import create_renderer

    datasets = []
    for data_dir in tqdm(sorted(data_root.iterdir()), desc="Loading datasets"):
        if not data_dir.is_dir():
            continue
        name = data_dir.name
        if "sample" in name:
            print(f"  [skip] {name}: sample dataset")
            continue

        info_path = data_dir / "info.json"
        gt_path   = data_dir / "gt_obj2cam.json"
        cam_path  = data_dir / "camera.json"
        est_path  = (results_root / name
                     / "foundpose-output" / "inference" / "estimated-poses.json")
        fit_path  = (results_root / name
                     / "fit-output" / "est-coarse-icp-colmap" / "fit-inliers.json")

        # Always-required files
        missing = [p.name for p in (info_path, gt_path, cam_path, est_path)
                   if not p.exists()]
        if missing:
            print(f"  [skip] {name}: missing {', '.join(missing)}")
            continue

        # BOP-only directories -- only checked when we need rendering
        if need_bop:
            depth_dir = data_dir / "depth-render"
            rgb_dir   = data_dir / "rgb"
            for d in (depth_dir, rgb_dir):
                if not d.exists():
                    missing.append(d.name)
            if missing:
                print(f"  [skip] {name}: missing {', '.join(missing)}")
                continue

        # -- always-loaded (cheap) -------------------------------------------
        info        = load_yaml(info_path)
        gt_poses    = load_yaml(gt_path)
        camera      = load_yaml(cam_path)
        est_poses   = load_json(est_path)
        fit_inliers = load_json(fit_path) if fit_path.exists() else None

        fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=float)

        # -- BOP-only (expensive) --------------------------------------------
        depth_paths = depth_imgs = None
        img_width = img_height = None
        renderer = vtxs = diameter = symTs = None

        if need_bop:
            depth_paths, depth_imgs_raw = imgs_from_dir(depth_dir, mode="I;16", asarray=True)
            depth_imgs = depth_imgs_raw / 1000.0      # mm -> m

            _, rgb_imgs = imgs_from_dir(rgb_dir)
            img_width, img_height = rgb_imgs[0].size

            if (img_width, img_height) not in resolution2renderer:
                ren = create_renderer(img_width, img_height,
                                      renderer_type="vispy", mode="depth")
                for mname in model_info_all:
                    ren.add_object(mname, model3d_root / mname)
                resolution2renderer[(img_width, img_height)] = ren
            renderer = resolution2renderer[(img_width, img_height)]
            renderer.set_current()

            object_name = info["object_name"]
            mesh: trimesh.Trimesh = trimesh.load(model3d_root / object_name)
            vtxs     = np.array(mesh.vertices)
            diameter = mesh.bounding_sphere.primitive.radius * 2
            symTs    = get_symmetry_transformations(model_info_all[object_name], 0.01)

        datasets.append(dict(
            name=name, info=info, est_poses=est_poses, gt_poses=gt_poses,
            fit_inliers=fit_inliers, K=K,
            # BOP-specific -- None when need_bop=False
            depth_paths=list(depth_paths) if depth_paths is not None else None,
            depth_imgs=depth_imgs,
            vtxs=vtxs, diameter=diameter, symTs=symTs, renderer=renderer,
            img_width=img_width, img_height=img_height,
            data_dir=data_dir, results_dir=results_root / name,
        ))

    mode = "full (BOP)" if need_bop else "lightweight (no BOP)"
    print(f"Loaded {len(datasets)} datasets ({mode}).")
    return datasets



# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _accepted_pairs(ds) -> set:
    fit = ds["fit_inliers"]
    pairs: set = set()
    if fit is None:
        return pairs
    iids = fit.get("img_ids", [])
    hids = fit.get("hyp_ids", [])
    for idx in fit.get("inlieridxs", []):
        if idx < len(iids):
            pairs.add((iids[idx], hids[idx]))
    return pairs

def _pnp_inlier_ratio(entry) -> float:
    dists   = entry.get("coord_match_dists", [])
    raw_inl = entry.get("ransac_inliers", [])
    return len(raw_inl) / len(dists) if len(dists) > 0 else np.nan

def _best_hyp_per_image(ds):
    accepted = _accepted_pairs(ds)
    img_hyps: dict = defaultdict(list)
    for e in ds["est_poses"]:
        img_hyps[stem(e["img_path"])].append(e)
    best: dict = {}
    for img_key, hyps in img_hyps.items():
        acc = [h for h in hyps if (h["img_id"], h["hypothesis_id"]) in accepted]
        best[img_key] = acc[0] if acc else max(hyps, key=lambda h: h["template_score"])
    return best, accepted


# ─────────────────────────────────────────────────────────────────────────────
# BOP metric computation  –  the slow part, shared by A6 and A8
# ─────────────────────────────────────────────────────────────────────────────

def compute_bop_rows(datasets, cache_dir: Path, overwrite: bool) -> list:
    """
    Compute (or load from cache) per-image BOP metric rows for every
    GT-labelled frame across all datasets.

    Each row is a dict with keys:
      dataset, burial_depth, burial_vol,
      pnp_inlier_ratio, template_score, is_fit_inlier,
      vsd, mssd, mspd,                          # DINOv2 coarse pose
      vsd_sift, mssd_sift, mspd_sift,           # SIFT pose (nan if unavailable)
      sift_valid,                                # bool
      n_imgs_in_dataset                          # used for A8a availability bar
    """
    if not overwrite:
        cached = _load_cache(cache_dir)
        if cached is not None:
            return cached

    print("\n  Computing BOP metrics (slow – will cache afterwards) ...")
    all_rows = []

    for ds in tqdm(datasets, desc="BOP eval"):
        bd   = float(ds["info"].get("burial_depth",     np.nan))
        bvol = float(ds["info"].get("burial_ratio_vol", np.nan))
        obj  = ds["info"].get("object_name", "unknown")

        gt_Rs    = np.array([gp["R"] for gp in ds["gt_poses"]])
        gt_ts    = np.array([gp["t"] for gp in ds["gt_poses"]])[..., None]
        gt_names = [stem(gp["img_path"]) for gp in ds["gt_poses"]]

        img_hyps_all: dict = defaultdict(list)
        for e in ds["est_poses"]:
            img_hyps_all[stem(e["img_path"])].append(e)

        accepted = _accepted_pairs(ds)

        for i, img_name in enumerate(gt_names):
            if img_name not in img_hyps_all:
                continue
            depth_idx = name_idx(img_name, ds["depth_paths"])
            if depth_idx < 0:
                continue

            hyps   = img_hyps_all[img_name]
            chosen = next((h for h in hyps
                           if str(h.get("hypothesis_id", "")) == "0"), hyps[0])

            R_c = np.array(chosen["R_coarse"])
            t_c = np.array(chosen["t_coarse"])
            vsd_c, mssd_c, mspd_c = _raw_bop_metrics(
                R_c, t_c, gt_Rs[i], gt_ts[i],
                ds["depth_imgs"][depth_idx], ds["K"],
                ds["renderer"], obj, ds["vtxs"], ds["diameter"],
                ds["img_width"], ds["symTs"],
            )
            is_inlier = (chosen["img_id"], chosen["hypothesis_id"]) in accepted

            # SIFT pose (may be None)
            r_sift_raw = chosen.get("R_sift", None)
            t_sift_raw = chosen.get("t_sift", None)
            sift_valid = (r_sift_raw is not None) and (t_sift_raw is not None)
            vsd_s = mssd_s = mspd_s = np.nan
            if sift_valid:
                R_sift = np.array(r_sift_raw)
                t_sift = np.array(t_sift_raw)
                vsd_s, mssd_s, mspd_s = _raw_bop_metrics(
                    R_sift, t_sift, gt_Rs[i], gt_ts[i],
                    ds["depth_imgs"][depth_idx], ds["K"],
                    ds["renderer"], obj, ds["vtxs"], ds["diameter"],
                    ds["img_width"], ds["symTs"],
                )

            all_rows.append(dict(
                dataset=ds["name"],
                burial_depth=bd, burial_vol=bvol,
                pnp_inlier_ratio=_pnp_inlier_ratio(chosen),
                template_score=float(chosen["template_score"]),
                is_fit_inlier=bool(is_inlier),
                vsd=vsd_c, mssd=mssd_c, mspd=mspd_c,
                vsd_sift=vsd_s, mssd_sift=mssd_s, mspd_sift=mspd_s,
                sift_valid=sift_valid,
            ))

    _save_cache(all_rows, cache_dir)
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4
# ─────────────────────────────────────────────────────────────────────────────

def analysis4_similarity_statistics(datasets, output_dir):
    print("\n[Analysis 4] Patch descriptor similarity statistics ...")
    scores_in, depths_in   = [], []
    scores_out, depths_out = [], []

    for ds in datasets:
        bd       = float(ds["info"].get("burial_depth", np.nan))
        accepted = _accepted_pairs(ds)
        img_hyps: dict = defaultdict(list)
        for e in ds["est_poses"]:
            img_hyps[stem(e["img_path"])].append(e)
        for hyps in img_hyps.values():
            acc = [h for h in hyps if (h["img_id"], h["hypothesis_id"]) in accepted]
            if acc:
                chosen = max(acc, key=lambda h: h["template_score"])
                scores_in.append(float(chosen["template_score"])); depths_in.append(bd)
            else:
                chosen = max(hyps, key=lambda h: h["template_score"])
                scores_out.append(float(chosen["template_score"])); depths_out.append(bd)

    if not (scores_in or scores_out):
        print("  No data – skipping."); return

    scores_in  = np.array(scores_in,  dtype=float)
    scores_out = np.array(scores_out, dtype=float)
    all_scores = np.concatenate([scores_in, scores_out])
    all_depths = np.concatenate([np.array(depths_in), np.array(depths_out)], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    bins = np.linspace(min(all_scores.min(), 0.0), all_scores.max(), 35)
    if len(scores_in):
        ax.hist(scores_in,  bins=bins, color=C_INLIER,  alpha=0.65, edgecolor="none",
                label=f"Fit-inlier  (N={len(scores_in)})")
    if len(scores_out):
        ax.hist(scores_out, bins=bins, color=C_OUTLIER, alpha=0.65, edgecolor="none",
                label=f"Fit-outlier (N={len(scores_out)})")
    ax.axvline(np.nanmean(all_scores),   color="black", lw=1.5, ls="--",
               label=f"Mean = {np.nanmean(all_scores):.3f}")
    ax.axvline(np.nanmedian(all_scores), color="black", lw=1.5, ls=":",
               label=f"Median = {np.nanmedian(all_scores):.3f}")
    ax.set_xlabel("Template cosine similarity score", fontsize=11)
    ax.set_ylabel("Count (images)", fontsize=11)
    ax.set_title("A4 – DINOv2 template similarity\nfit-inliers vs fit-outliers", fontsize=11)
    ax.legend(fontsize=8)

    valid = ~np.isnan(all_depths)
    if valid.sum() > 4:
        ax = axes[1]
        qs = np.nanpercentile(all_depths[valid], [25, 50, 75])
        qlabels = [f"Q1 ≤{qs[0]:.2f} m", f"Q2 ≤{qs[1]:.2f} m",
                   f"Q3 ≤{qs[2]:.2f} m", f"Q4 >{qs[2]:.2f} m"]
        bins_q = [-np.inf, qs[0], qs[1], qs[2], np.inf]
        groups = [all_scores[valid & (all_depths >= bins_q[i]) & (all_depths < bins_q[i+1])]
                  for i in range(4)]
        groups = [g[~np.isnan(g)] for g in groups]
        bp = ax.violinplot(groups, positions=range(4), showmedians=True, showextrema=True)
        for pc in bp["bodies"]:
            pc.set_facecolor(C_BOTH); pc.set_alpha(0.65)
        ax.set_xticks(range(4)); ax.set_xticklabels(qlabels, fontsize=8)
        ax.set_ylabel("Template cosine similarity score", fontsize=11)
        ax.set_title("A4 – Similarity by burial depth quartile", fontsize=11)
        for i, g in enumerate(groups):
            if len(g):
                ax.text(i, g.max() + 0.005, f"μ={g.mean():.3f}",
                        ha="center", fontsize=7, color="navy")
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "A4_similarity_stats.pdf")
    print(f"  Overall: mean={np.nanmean(all_scores):.4f}  std={np.nanstd(all_scores):.4f}  N={len(all_scores)}")
    if len(scores_in):
        print(f"    Inlier : mean={scores_in.mean():.4f}  std={scores_in.std():.4f}  N={len(scores_in)}")
    if len(scores_out):
        print(f"    Outlier: mean={scores_out.mean():.4f}  std={scores_out.std():.4f}  N={len(scores_out)}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4b
# ─────────────────────────────────────────────────────────────────────────────

def analysis4b_feature_distances(datasets, output_dir):
    print("\n[Analysis 4b] Feature distance distributions ...")
    dists_in, dists_out = [], []
    for ds in datasets:
        accepted = _accepted_pairs(ds)
        for e in ds["est_poses"]:
            d = e.get("coord_match_dists", [])
            if not d:
                continue
            if (e["img_id"], e["hypothesis_id"]) in accepted:
                dists_in.extend(d)
            else:
                dists_out.extend(d)

    if not (dists_in or dists_out):
        print("  No distance data – skipping."); return

    dists_in  = np.array(dists_in,  dtype=float)
    dists_out = np.array(dists_out, dtype=float)
    all_d = np.concatenate([dists_in, dists_out])
    hi    = np.percentile(all_d, 99)
    bins  = np.linspace(0, hi, 60)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    if len(dists_in):
        ax.hist(dists_in,  bins=bins, color=C_INLIER,  alpha=0.65, edgecolor="none",
                label=f"Fit-inlier  (N={len(dists_in):,})")
    if len(dists_out):
        ax.hist(dists_out, bins=bins, color=C_OUTLIER, alpha=0.65, edgecolor="none",
                label=f"Fit-outlier (N={len(dists_out):,})")
    ax.set_yscale("log")
    ax.set_xlabel("Correspondence distance norm", fontsize=11)
    ax.set_ylabel("Count (log scale)", fontsize=11)
    ax.set_title("A4b – DINOv2 match distances\nfit-inliers vs fit-outliers", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    for arr, color, label in [
        (dists_in,  C_INLIER,  f"Fit-inlier  (N={len(dists_in):,})"),
        (dists_out, C_OUTLIER, f"Fit-outlier (N={len(dists_out):,})"),
    ]:
        if len(arr):
            s = np.sort(arr); cdf = np.arange(1, len(s)+1) / len(s)
            ax.plot(s, cdf, color=color, lw=1.8, label=label)
    ax.set_xlim(left=0)
    ax.set_xlabel("Correspondence distance norm", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title("A4b – CDF of feature match distances", fontsize=11)
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, output_dir, "A4b_feature_distances.pdf")
    if len(dists_in):
        print(f"  Inlier  dists: mean={dists_in.mean():.3f}  median={np.median(dists_in):.3f}  N={len(dists_in):,}")
    if len(dists_out):
        print(f"  Outlier dists: mean={dists_out.mean():.3f}  median={np.median(dists_out):.3f}  N={len(dists_out):,}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 5
# ─────────────────────────────────────────────────────────────────────────────

def analysis5_inlier_ratio(datasets, output_dir):
    print("\n[Analysis 5] Inlier ratio analysis ...")
    pnp_rows = []; fit_rows = []

    for ds in datasets:
        bd   = float(ds["info"].get("burial_depth",     np.nan))
        bvol = float(ds["info"].get("burial_ratio_vol", np.nan))
        accepted = _accepted_pairs(ds)
        img_hyps: dict = defaultdict(list)
        for e in ds["est_poses"]:
            img_hyps[stem(e["img_path"])].append(e)
        for hyps in img_hyps.values():
            acc    = [h for h in hyps if (h["img_id"], h["hypothesis_id"]) in accepted]
            chosen = acc[0] if acc else max(hyps, key=lambda h: h["template_score"])
            is_in  = (chosen["img_id"], chosen["hypothesis_id"]) in accepted
            pnp_rows.append(dict(burial_depth=bd, burial_vol=bvol,
                                 pnp_inlier_ratio=_pnp_inlier_ratio(chosen),
                                 template_score=float(chosen["template_score"]),
                                 is_fit_inlier=is_in))
        fit = ds["fit_inliers"]
        if fit is not None:
            n_total = len(fit.get("img_ids", []))
            if n_total > 0:
                fit_rows.append(dict(burial_depth=bd, burial_vol=bvol,
                                     hyp_inlier_ratio=len(fit.get("inlieridxs", [])) / n_total,
                                     dataset=ds["name"]))

    if pnp_rows:
        depths = np.array([r["burial_depth"]     for r in pnp_rows], dtype=float)
        vols   = np.array([r["burial_vol"]       for r in pnp_rows], dtype=float)
        y      = np.array([r["pnp_inlier_ratio"] for r in pnp_rows], dtype=float)
        scores = np.array([r["template_score"]   for r in pnp_rows], dtype=float)
        is_in  = np.array([r["is_fit_inlier"]    for r in pnp_rows], dtype=bool)
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        _scatter_inout(axes[0], depths, y, is_in, "Burial depth (m)", "PnP+RANSAC inlier ratio",
                       "A5a – PnP inlier ratio vs burial depth")
        _scatter_inout(axes[1], vols,   y, is_in, "Burial volume ratio", "PnP+RANSAC inlier ratio",
                       "A5a – PnP inlier ratio vs burial volume ratio")
        _scatter_inout(axes[2], scores, y, is_in, "Template cosine similarity", "PnP+RANSAC inlier ratio",
                       "A4↔A5a – Template score vs PnP inlier ratio")
        fig.tight_layout(); _save(fig, output_dir, "A5a_pnp_inlier_ratio.pdf")
        print(f"  A5a PnP IR: mean={np.nanmean(y):.4f}  std={np.nanstd(y):.4f}  N={int(np.sum(~np.isnan(y)))}")

    if fit_rows:
        depths = np.array([r["burial_depth"]     for r in fit_rows], dtype=float)
        vols   = np.array([r["burial_vol"]       for r in fit_rows], dtype=float)
        y      = np.array([r["hyp_inlier_ratio"] for r in fit_rows], dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        _scatter_reg(axes[0], depths, y, C_GREEN, "Burial depth (m)",
                     "Hypothesis acceptance rate\n(# RANSAC-kept / total hypotheses per dataset)",
                     "A5b – Dataset-level hyp acceptance rate\nvs burial depth")
        _scatter_reg(axes[1], vols,   y, C_GREEN, "Burial volume ratio",
                     "Hypothesis acceptance rate\n(# RANSAC-kept / total hypotheses per dataset)",
                     "A5b – Dataset-level hyp acceptance rate\nvs burial volume ratio")
        fig.tight_layout(); _save(fig, output_dir, "A5b_hyp_inlier_ratio.pdf")
        print(f"  A5b Hyp acceptance: mean={np.nanmean(y):.4f}  std={np.nanstd(y):.4f}  N={int(np.sum(~np.isnan(y)))} datasets")
    else:
        print("  A5b: no fit-inliers data found – skipping.")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 6
# ─────────────────────────────────────────────────────────────────────────────

def analysis6_bop_metrics(all_rows, output_dir):
    print("\n[Analysis 6] Plotting BOP pose metrics ...")
    if not all_rows:
        print("  No rows – skipping."); return

    bop_triples = [
        ("vsd",  "VSD", "#4C72B0"),
        ("mssd", "MSSD (m)",  "#55A868"),
        ("mspd", "MSPD (pixels)",     "#DD8452"),
    ]

    dep_all   = np.array([r["burial_depth"]    for r in all_rows], dtype=float)
    vol_all   = np.array([r["burial_vol"]      for r in all_rows], dtype=float)
    ir_all    = np.array([r["pnp_inlier_ratio"]for r in all_rows], dtype=float)
    is_in_arr = np.array([r["is_fit_inlier"]   for r in all_rows], dtype=bool)

    # ── A6a ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    for row_i, (col, label, color) in enumerate(bop_triples):
        vals = np.array([r[col] for r in all_rows], dtype=float)
        _scatter_reg(axes[row_i][0], dep_all, vals, color,
                     "Burial depth (m)", label, f"A6 – {label}\nvs burial depth")
        _scatter_reg(axes[row_i][1], vol_all, vals, color,
                     "Burial volume ratio", label, f"A6 – {label}\nvs burial volume ratio")
        _scatter_inout(axes[row_i][2], dep_all, vals, is_in_arr,
                       "Burial depth (m)", label, f"A6 – {label} vs depth\n(fit-inlier coloured)")
        _scatter_inout(axes[row_i][3], ir_all, vals, is_in_arr,
                       "PnP+RANSAC inlier ratio", label, f"A5↔A6 – inlier ratio vs {label}")
    fig.tight_layout(); _save(fig, output_dir, "A6a_bop_vs_burial.pdf")

    # ── A6b ──────────────────────────────────────────────────────────────────
    feature_pairs = [
        ("pnp_inlier_ratio", "PnP+RANSAC inlier ratio"),
        ("template_score",   "Template cosine similarity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row_i, (feat_col, feat_label) in enumerate(feature_pairs):
        x = np.array([r[feat_col] for r in all_rows], dtype=float)
        print(f"number of valid {feat_label} values: {(~np.isnan(x)).sum()} / {len(x)}")
        for col_i, (bop_col, bop_label, bop_color) in enumerate(bop_triples):
            if bop_col == "vsd":
                continue
            y  = np.array([r[bop_col] for r in all_rows], dtype=float)
            ax = axes[row_i][col_i - 1]

            valid  = ~(np.isnan(x) | np.isnan(y))
            ax.scatter(x[valid], y[valid], c="tab:blue", alpha=0.35,
                       s=14, edgecolors="none", zorder=2)

            order = np.argsort(x[valid])
            x_s, y_s = x[valid][order], y[valid][order]
            xg, ym, yq25, yq75 = _rolling_stats(x_s, y_s)
            good = ~(np.isnan(ym) | np.isnan(yq25) | np.isnan(yq75))
            if good.sum() > 1:
                ax.plot(xg[good], ym[good], color="black", lw=2.0, zorder=4,
                        label="Rolling median")
                ax.fill_between(xg[good], yq25[good], yq75[good],
                                color="black", alpha=0.15, zorder=3,
                                label="Rolling Q25–Q75")

            ax.legend(handles=[
                plt.Line2D([0], [0], color="black", lw=2, label="Rolling median"),
                mpatches.Patch(color="black", alpha=0.2, label="Rolling Q25–Q75"),
            ], fontsize=7)
            ax.tick_params(axis="both", labelsize=14)
            ax.set_title(f"A6b – {feat_label}\nvs {bop_label}", fontsize=18)
            ax.set_xlabel(feat_label, fontsize=18)
            ax.set_ylabel(bop_label,  fontsize=18)
    fig.tight_layout(); _save(fig, output_dir, "A6b_feature_vs_bop.pdf")

    # ── A6c ──────────────────────────────────────────────────────────────────
    valid_d = ~np.isnan(dep_all)
    if valid_d.sum() > 4:
        qs      = np.nanpercentile(dep_all[valid_d], [25, 50, 75])
        qlabels = [f"Q1 ≤{qs[0]:.2f} m", f"Q2 ≤{qs[1]:.2f} m",
                   f"Q3 ≤{qs[2]:.2f} m", f"Q4 >{qs[2]:.2f} m"]
        bins_q  = [-np.inf, qs[0], qs[1], qs[2], np.inf]
        fig, axes = plt.subplots(3, 1, figsize=(5, 9))
        for col_i, (bop_col, bop_label, color) in enumerate(bop_triples):
            vals   = np.array([r[bop_col] for r in all_rows], dtype=float)
            groups = [vals[valid_d & (dep_all >= bins_q[i]) & (dep_all < bins_q[i+1])]
                      for i in range(4)]
            print(f"  A6c {bop_label} by depth quartile: group sizes {[len(g) for g in groups]}")
            groups = [g[~np.isnan(g)] for g in groups]
            ax = axes[col_i]
            bp = ax.violinplot(groups, positions=range(4), showmedians=True, showextrema=True)
            for pc in bp["bodies"]:
                # pc.set_facecolor(color); pc.set_alpha(0.65)
                pc.set_facecolor("#DD8452"); pc.set_alpha(0.65)
            ax.set_xticks(range(4)); ax.set_xticklabels(qlabels, fontsize=7)
            ax.tick_params(axis="both", labelsize=7)
            ax.set_ylabel(bop_label, fontsize=9)
            ax.set_title(f"A6c – {bop_label}\nby burial depth quartile", fontsize=9)
            for i, g in enumerate(groups):
                if len(g):
                    ax.text(i, np.nanmax(g) * 1.02, f"μ={np.nanmean(g):.3f}",
                            ha="center", fontsize=7, color="navy")
        fig.tight_layout(); _save(fig, output_dir, "A6c_bop_by_depth_quartile.pdf")

    for col, label, _ in bop_triples:
        v = np.array([r[col] for r in all_rows], dtype=float)
        print(f"  {label}: mean={np.nanmean(v):.4f}  std={np.nanstd(v):.4f}  "
              f"median={np.nanmedian(v):.4f}  N={int(np.sum(~np.isnan(v)))}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 7
# ─────────────────────────────────────────────────────────────────────────────

def analysis7_template_ranking(datasets, output_dir):
    print("\n[Analysis 7] Template ranking analysis ...")
    records = []
    for ds in datasets:
        bd  = float(ds["info"].get("burial_depth", np.nan))
        fit = ds["fit_inliers"]
        if fit is None:
            continue
        iids = fit.get("img_ids", [])
        hids = fit.get("hyp_ids", [])
        accepted: set = {(iids[i], hids[i]) for i in fit.get("inlieridxs", []) if i < len(iids)}
        by_img: dict = defaultdict(list)
        for e in ds["est_poses"]:
            by_img[e["img_id"]].append((e["hypothesis_id"], float(e["template_score"])))
        for img_id, hyps in by_img.items():
            hyps_sorted = sorted(hyps, key=lambda x: -x[1])
            acc = [(h, s) for h, s in hyps_sorted if (img_id, h) in accepted]
            if acc:
                chosen_hyp = acc[0][0]; selected_by_ransac = True
            else:
                chosen_hyp = hyps_sorted[0][0]; selected_by_ransac = False
            rank = next((i + 1 for i, (h, _) in enumerate(hyps_sorted) if h == chosen_hyp), None)
            records.append(dict(burial_depth=bd, chosen_rank=rank,
                                n_hyps=len(hyps), selected_by_ransac=selected_by_ransac))

    if not records:
        print("  No fit-inliers data – skipping."); return

    ranks    = np.array([r["chosen_rank"]       for r in records if r["chosen_rank"] is not None])
    depths   = np.array([r["burial_depth"]       for r in records if r["chosen_rank"] is not None], dtype=float)
    selected = np.array([r["selected_by_ransac"] for r in records if r["chosen_rank"] is not None])
    n_hyps_max = max(r["n_hyps"] for r in records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    bins = np.arange(0.5, n_hyps_max + 1.5)
    ax.hist(ranks[selected],  bins=bins, color=C_INLIER, alpha=0.80, edgecolor="white",
            label=f"RANSAC-accepted (N={selected.sum()})")
    ax.hist(ranks[~selected], bins=bins, color=C_BOTH,   alpha=0.55, edgecolor="white",
            label=f"Fallback top-1  (N={(~selected).sum()})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Score rank of chosen hypothesis\n(1 = highest cosine similarity)", fontsize=11)
    ax.set_ylabel("Count (images)", fontsize=11)
    ax.set_title("A7 – Score rank of RANSAC-selected hypothesis", fontsize=11)
    ax.legend(fontsize=8)

    ax = axes[1]
    valid  = ~np.isnan(depths)
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, valid.sum())
    colors = np.where(selected[valid], C_INLIER, C_BOTH)
    ax.scatter(depths[valid], ranks[valid] + jitter, c=colors, alpha=0.55, s=16, edgecolors="none")
    ax.legend(handles=[mpatches.Patch(color=C_INLIER, label="RANSAC-accepted"),
                       mpatches.Patch(color=C_BOTH,   label="Fallback top-1")], fontsize=8)
    if valid.sum() > 2:
        m, b, r_v, p, _ = stats.linregress(depths[valid], ranks[valid])
        xl = np.linspace(np.nanmin(depths[valid]), np.nanmax(depths[valid]), 100)
        ax.plot(xl, m * xl + b, "black", lw=1.5, label=f"r = {r_v:.3f},  p = {p:.3g}")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Burial depth (m)", fontsize=11)
    ax.set_ylabel("Score rank of chosen hypothesis", fontsize=11)
    ax.set_title("A7 – Chosen hypothesis rank vs burial depth", fontsize=11)

    fig.tight_layout(); _save(fig, output_dir, "A7_template_ranking.pdf")
    print(f"  Top-1 rank: {(ranks == 1).mean()*100:.1f}%  mean rank: {ranks.mean():.2f} ± {ranks.std():.2f}")
    print(f"  RANSAC-accepted: {selected.sum()} / {len(selected)} ({selected.mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 8
# ─────────────────────────────────────────────────────────────────────────────

def analysis8_sift_vs_dinov2(all_rows, output_dir):
    print("\n[Analysis 8] SIFT vs DINOv2 BOP comparison ...")
    if not all_rows:
        print("  No rows – skipping."); return

    bop_triples = [
        ("vsd",  "VSD", "#4C72B0"),
        ("mssd", "MSSD (m)",  "#55A868"),
        ("mspd", "MSPD (pixels)",     "#DD8452"),
    ]

    rows   = all_rows
    paired = [r for r in rows if r["sift_valid"]]

    # ── A8a ──────────────────────────────────────────────────────────────────
    # availability per dataset
    by_ds: dict = defaultdict(lambda: {"total": 0, "sift": 0, "bd": np.nan})
    for r in rows:
        by_ds[r["dataset"]]["total"] += 1
        if r["sift_valid"]:
            by_ds[r["dataset"]]["sift"] += 1
        by_ds[r["dataset"]]["bd"] = r["burial_depth"]

    dataset_avail = sorted(
        [(ds, d["bd"], d["sift"] / d["total"] if d["total"] else 0.0)
         for ds, d in by_ds.items()],
        key=lambda x: x[1] if not np.isnan(x[1]) else 1e9,
    )
    if dataset_avail:
        ds_labels   = [d[0] for d in dataset_avail]
        avail_fracs = np.array([d[2] for d in dataset_avail])
        burial_deps = np.array([d[1] for d in dataset_avail], dtype=float)
        fig, ax = plt.subplots(figsize=(max(8, len(ds_labels) * 0.55), 4.8))
        bars = ax.bar(range(len(ds_labels)), avail_fracs * 100,
                      color=C_GREEN, edgecolor="white", alpha=0.8)
        norm_bd = ((burial_deps - np.nanmin(burial_deps)) /
                   (np.nanmax(burial_deps) - np.nanmin(burial_deps) + 1e-9))
        cmap = plt.cm.YlOrRd
        for bar, nb in zip(bars, norm_bd):
            bar.set_facecolor(cmap(nb))
        ax.set_xticks(range(len(ds_labels)))
        ax.set_xticklabels(ds_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Images with valid SIFT pose (%)", fontsize=11)
        ax.set_title("A8a – SIFT pose availability per dataset\n(ordered by burial depth, colour = depth)", fontsize=11)
        ax.set_ylim(0, 105)
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=np.nanmin(burial_deps),
                                                      vmax=np.nanmax(burial_deps)))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Burial depth (m)", pad=0.01)
        fig.tight_layout(); _save(fig, output_dir, "A8a_sift_availability.pdf")
        print(f"  SIFT available: {len(paired)} / {len(rows)} images ({100*len(paired)/len(rows):.1f}%)")

    # ── A8b ──────────────────────────────────────────────────────────────────
    if paired:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for col_i, (bop_col, bop_label, color) in enumerate(bop_triples):
            x = np.array([r[f"{bop_col}"]      for r in paired], dtype=float)
            y = np.array([r[f"{bop_col}_sift"]  for r in paired], dtype=float)
            valid = ~(np.isnan(x) | np.isnan(y))
            ax = axes[col_i]
            ax.scatter(x[valid], y[valid], color=color, alpha=0.55, s=18, edgecolors="none")
            all_vals = np.concatenate([x[valid], y[valid]])
            lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Identity (equal)")
            n_dino_better = (x[valid] < y[valid]).sum()
            n_sift_better = (y[valid] < x[valid]).sum()
            ax.set_xlabel(f"DINOv2 {bop_label}", fontsize=10)
            ax.set_ylabel(f"SIFT {bop_label}",   fontsize=10)
            ax.set_title(f"A8b – {bop_label}\nDINOv2 better: {n_dino_better}  "
                         f"SIFT better: {n_sift_better}  (N={valid.sum()})", fontsize=9)
            ax.legend(fontsize=8)
        fig.tight_layout(); _save(fig, output_dir, "A8b_sift_vs_dinov2_bop.pdf")

        for bop_col, bop_label, _ in bop_triples:
            x = np.array([r[bop_col]           for r in paired], dtype=float)
            y = np.array([r[f"{bop_col}_sift"]  for r in paired], dtype=float)
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() == 0: continue
            n_better = (x[valid] < y[valid]).sum()
            print(f"    {bop_label}: DINOv2 better in {n_better}/{valid.sum()} "
                  f"({100*n_better/valid.sum():.1f}%)  "
                  f"median DINOv2={np.nanmedian(x):.4f}  median SIFT={np.nanmedian(y):.4f}")
    else:
        print("  No images with valid SIFT pose – skipping A8b.")

    # ── A8c ──────────────────────────────────────────────────────────────────
    if paired:
        dep_paired = np.array([r["burial_depth"] for r in paired], dtype=float)
        valid_d    = ~np.isnan(dep_paired)
        if valid_d.sum() > 4:
            qs      = np.nanpercentile(dep_paired[valid_d], [25, 50, 75])
            qlabels = [f"Q1 ≤{qs[0]:.2f} m", f"Q2 ≤{qs[1]:.2f} m",
                       f"Q3 ≤{qs[2]:.2f} m", f"Q4 >{qs[2]:.2f} m"]
            bins_q  = [-np.inf, qs[0], qs[1], qs[2], np.inf]
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            for col_i, (bop_col, bop_label, _) in enumerate(bop_triples):
                x_d = np.array([r[bop_col]          for r in paired], dtype=float)
                x_s = np.array([r[f"{bop_col}_sift"] for r in paired], dtype=float)
                ax  = axes[col_i]
                pos_d, pos_s, grp_d, grp_s, tick_pos, tick_lab = [], [], [], [], [], []
                for qi in range(4):
                    mask = valid_d & (dep_paired >= bins_q[qi]) & (dep_paired < bins_q[qi+1])
                    base = qi * 3
                    pos_d.append(base); pos_s.append(base + 1)
                    grp_d.append(x_d[mask & ~np.isnan(x_d)])
                    grp_s.append(x_s[mask & ~np.isnan(x_s)])
                    tick_pos.append(base + 0.5); tick_lab.append(qlabels[qi])
                gd_nonempty = [g for g in grp_d if len(g)]
                gs_nonempty = [g for g in grp_s if len(g)]
                pd_nonempty = [pos_d[i] for i, g in enumerate(grp_d) if len(g)]
                ps_nonempty = [pos_s[i] for i, g in enumerate(grp_s) if len(g)]
                if gd_nonempty:
                    vp = ax.violinplot(gd_nonempty, positions=pd_nonempty, showmedians=True, showextrema=True)
                    for pc in vp["bodies"]: pc.set_facecolor(C_INLIER); pc.set_alpha(0.65)
                if gs_nonempty:
                    vp = ax.violinplot(gs_nonempty, positions=ps_nonempty, showmedians=True, showextrema=True)
                    for pc in vp["bodies"]: pc.set_facecolor(C_ORANGE); pc.set_alpha(0.65)
                ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, fontsize=8)
                ax.set_ylabel(bop_label, fontsize=10)
                ax.set_title(f"A8c – {bop_label}\nDINOv2 vs SIFT by depth quartile", fontsize=9)
                ax.legend(handles=[mpatches.Patch(color=C_INLIER,  label="DINOv2"),
                                   mpatches.Patch(color=C_ORANGE, label="SIFT")], fontsize=8)
            fig.tight_layout(); _save(fig, output_dir, "A8c_bop_by_method_depth.pdf")
        else:
            print("  Too few paired data points for quartile plot (A8c) – skipping.")

    # ── A8d ──────────────────────────────────────────────────────────────────
    if paired:
        dep_p = np.array([r["burial_depth"] for r in paired], dtype=float)
        OUTLIER_THRESH_FACTOR = 5.0
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for col_i, (bop_col, bop_label, color) in enumerate(bop_triples):
            ax = axes[col_i]
            y_d = np.array([r[bop_col]           for r in paired], dtype=float)
            y_s = np.array([r[f"{bop_col}_sift"]  for r in paired], dtype=float)
            valid_d = ~(np.isnan(dep_p) | np.isnan(y_d))
            valid_s = ~(np.isnan(dep_p) | np.isnan(y_s))

            n_out = 0; thresh = np.inf; y_cap_hi = np.inf
            if valid_s.sum() > 4:
                q1_s, q3_s = np.nanpercentile(y_s[valid_s], [25, 75])
                iqr_s  = q3_s - q1_s
                thresh = q3_s + OUTLIER_THRESH_FACTOR * iqr_s
                y_cap_hi = q3_s + 3.0 * iqr_s
                n_out = int((valid_s & (y_s > thresh)).sum())
            y_d_max = np.nanmax(y_d[valid_d]) if valid_d.sum() else 0.0
            ymax = max(y_cap_hi if np.isfinite(y_cap_hi) else 0.0, y_d_max) * 1.05
            ymin = 0.0

            ax.scatter(dep_p[valid_d], y_d[valid_d],
                       color=C_INLIER, alpha=0.65, s=22, edgecolors="none",
                       label=f"DINOv2 (N={valid_d.sum()})", zorder=3)
            ax.scatter(dep_p[valid_s], y_s[valid_s],
                       color=C_ORANGE, alpha=0.55, s=22, edgecolors="none",
                       label=f"SIFT    (N={valid_s.sum()})", zorder=2)
            for yvals, valid, c, ls in [(y_d, valid_d, C_INLIER, "-"),
                                        (y_s, valid_s, C_ORANGE, "--")]:
                if valid.sum() > 2:
                    m, b, r_v, p_v, _ = stats.linregress(dep_p[valid], yvals[valid])
                    xl = np.linspace(np.nanmin(dep_p[valid]), np.nanmax(dep_p[valid]), 200)
                    yl = m * xl + b
                    in_window = (yl >= ymin) & (yl <= ymax)
                    if in_window.any():
                        ax.plot(xl[in_window], yl[in_window], color=c, lw=1.4, ls=ls,
                                label=f"r={r_v:.3f} p={p_v:.3g}")
            if n_out > 0:
                outlier_mask = valid_s & (y_s > thresh)
                ax.scatter(dep_p[outlier_mask], np.full(n_out, ymax * 0.97),
                           color="red", s=70, marker="x", lw=2.0, zorder=5,
                           label=f"SIFT outliers clipped (N={n_out}, >{thresh:.2f})")
                ax.text(0.98, 0.97,
                        f"{n_out} SIFT point(s) > {thresh:.2f}\nclipped — shown at top edge",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7, color="red",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("Burial depth (m)", fontsize=10)
            ax.set_ylabel(bop_label, fontsize=10)
            ax.set_title(f"A8d – {bop_label}: DINOv2 vs SIFT\nvs burial depth", fontsize=10)
            ax.legend(fontsize=7)
        fig.tight_layout(); _save(fig, output_dir, "A8d_depth_vs_bop_dino_vs_sift.pdf")

    # ── A8e ──────────────────────────────────────────────────────────────────
    sift_rows = [r for r in rows if r["sift_valid"]]
    if sift_rows:
        fig, axes = plt.subplots(3, 1, figsize=(5, 9))
        for col_i, (bop_col, bop_label, _) in enumerate(bop_triples):
            ax = axes[col_i]
            y_d_all = np.array([r[bop_col]           for r in rows],      dtype=float)
            y_s     = np.array([r[f"{bop_col}_sift"]  for r in sift_rows], dtype=float)
            y_d_all = y_d_all[~np.isnan(y_d_all)]
            y_s     = y_s[~np.isnan(y_s)]
            if not (len(y_d_all) and len(y_s)):
                ax.set_visible(False); continue

            combined = np.concatenate([y_d_all, y_s])
            q1_c, q3_c = np.percentile(combined, [25, 75])
            fence = 1.0 if bop_col == "vsd" else q3_c + 3.0 * (q3_c - q1_c)
            n_d_dropped = int((y_d_all > fence).sum())
            n_s_dropped = int((y_s     > fence).sum())
            y_d_filt    = y_d_all[y_d_all <= fence]
            y_s_filt    = y_s[y_s <= fence]
            print(f"number of filtered values: {bop_label}: DINOv2 filtered {len(y_d_filt)} / {len(y_d_all)} ")
            print(f"number of filtered values: {bop_label}: SIFT filtered {len(y_s_filt)} / {len(y_s)} ")
            if not (len(y_d_filt) and len(y_s_filt)):
                ax.set_visible(False); continue

            bins = np.linspace(0, fence, 20)
            ax.hist(y_d_filt, bins=bins, color=C_INLIER, alpha=0.65, density=True,
                edgecolor="none",
                label=f"DINOv2 all (N={len(y_d_filt)})"
                # label=f"DINOv2"
            )
            ax.hist(y_s_filt, bins=bins, color=C_ORANGE, alpha=0.65, density=True,
                edgecolor="none",
                label=f"SIFT valid (N={len(y_s_filt)})"
                # label=f"SIFT"
            )
            ax.tick_params(axis="both", labelsize=7)
            ax.set_xlabel(bop_label, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            # ax.set_title(f"A8e – {bop_label} distribution\nDINOv2 (all) vs SIFT (valid only)", fontsize=10)
            ax.set_title(f"{bop_label}", fontsize=10)
            ax.legend(fontsize=7)
        fig.tight_layout(); _save(fig, output_dir, "A8e_bop_histograms_dino_vs_sift.pdf")
    else:
        print("  No images with valid SIFT pose – skipping A8e.")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def summary_correlation_table(datasets, bop_rows, output_dir):
    print("\n[Summary] Correlation table ...")
    col_score, col_pnp_ir, col_depth = [], [], []
    for ds in datasets:
        bd = float(ds["info"].get("burial_depth", np.nan))
        best, _ = _best_hyp_per_image(ds)
        for e in best.values():
            col_score.append(float(e["template_score"]))
            col_pnp_ir.append(_pnp_inlier_ratio(e))
            col_depth.append(bd)

    col_score  = np.array(col_score)
    col_pnp_ir = np.array(col_pnp_ir)
    col_depth  = np.array(col_depth, dtype=float)

    table_rows = []
    for label, vals in [("Template cosine similarity", col_score),
                        ("PnP+RANSAC inlier ratio",    col_pnp_ir)]:
        valid = ~(np.isnan(vals) | np.isnan(col_depth))
        r_v, p_v = (stats.pearsonr(col_depth[valid], vals[valid])
                    if valid.sum() > 2 else (np.nan, np.nan))
        table_rows.append((label, np.nanmean(vals), np.nanstd(vals), r_v, p_v, int(valid.sum())))

    if bop_rows:
        for label, col in [("VSD (coarse)", "vsd"), ("MSSD (coarse)", "mssd"), ("MSPD (coarse)", "mspd")]:
            vals   = np.array([r[col]            for r in bop_rows], dtype=float)
            depths = np.array([r["burial_depth"] for r in bop_rows], dtype=float)
            valid  = ~(np.isnan(vals) | np.isnan(depths))
            if valid.sum() < 3: continue
            r_v, p_v = stats.pearsonr(depths[valid], vals[valid])
            table_rows.append((label, np.nanmean(vals), np.nanstd(vals), r_v, p_v, int(valid.sum())))

    print(f"\n  {'Metric':<35}  {'Mean':>8}  {'Std':>8}  {'r':>8}  {'p':>10}  N")
    print("  " + "-" * 80)
    for name, mu, sd, r_v, p_v, n in table_rows:
        r_s = f"{r_v:+.3f}" if not np.isnan(r_v) else "--"
        p_s = f"{p_v:.3g}"  if not np.isnan(p_v) else "--"
        print(f"  {name:<35}  {mu:>8.4f}  {sd:>8.4f}  {r_s:>8}  {p_s:>10}  {n}")

    latex = (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{DINOv2 feature-level and BOP pose-error metrics. "
        "Pearson $r$ vs burial depth (m). "
        "VSD $\\in[0,1]$: mean surface discrepancy; "
        "MSSD: max symmetric surface distance (CAD units); "
        "MSPD: max symmetric projected distance (px). "
        "Lower is better for all metrics.}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Metric & Mean & Std & $r$ vs depth & $p$ \\\\\n\\midrule\n"
    )
    for name, mu, sd, r_v, p_v, n in table_rows:
        r_s = f"{r_v:+.3f}" if not np.isnan(r_v) else "--"
        p_s = f"{p_v:.2e}"  if not np.isnan(p_v) else "--"
        latex += f"{name} & {mu:.4f} & {sd:.4f} & {r_s} & {p_s} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    out = Path(output_dir) / "table_feature_correlations.tex"
    out.write_text(latex)
    print(f"\n  LaTeX table -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 reliability analysis for underwater 6DoF pose estimation")
    parser.add_argument("--data_root",
                        default="/Volumes/T7/backup/Projects/ms-stuff/barrel-playground/barrels/data/input_data",
                        help="ROOT_DATA_DIR: parent of <dataset>/info.json, …")
    parser.add_argument("--results_root",
                        default="/Volumes/T7/backup/Projects/ms-stuff/barrel-playground/barrels/results",
                        help="ROOT_RESULTS_DIR: parent of <dataset>/foundpose-output/ and fit-output/")
    parser.add_argument("--model3d_root",
                        default="/Volumes/T7/backup/Projects/ms-stuff/barrel-playground/models3d",
                        help="Directory containing *.ply CAD models and model_info.json")
    parser.add_argument("--output_dir",  default="results/figures",
                        help="Output directory for figures and tables (default: results/figures)")
    parser.add_argument("--cache_dir",   default="/Volumes/T7/backup/Projects/ms-stuff/barrel-playground/barrels/results",
                        help="Directory for bop_cache.json (default: results)")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Recompute BOP metrics even if a cache file exists")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DINOv2 Reliability Analysis")
    print("=" * 60)

    # Decide whether heavy BOP infrastructure is needed.
    # Skip it entirely when a valid cache exists and --overwrite is not set.
    cache_dir = Path(args.cache_dir)
    need_bop  = args.overwrite or _load_cache(cache_dir) is None
    if not need_bop:
        print("  Cache found - loading datasets in lightweight mode "
              "(depth images / renderers / meshes skipped).")

    datasets = discover_datasets(args.data_root, args.results_root,
                                 args.model3d_root, need_bop=need_bop)
    if not datasets:
        print("ERROR: No datasets loaded.")
        return

    # -- non-BOP analyses (fast, always recomputed) --------------------------
    analysis4_similarity_statistics(datasets, args.output_dir)
    analysis4b_feature_distances(datasets, args.output_dir)
    analysis5_inlier_ratio(datasets, args.output_dir)
    analysis7_template_ranking(datasets, args.output_dir)

    # -- BOP analyses (slow - cached) ----------------------------------------
    bop_rows = compute_bop_rows(datasets, cache_dir, args.overwrite)

    analysis6_bop_metrics(bop_rows, args.output_dir)
    analysis8_sift_vs_dinov2(bop_rows, args.output_dir)
    summary_correlation_table(datasets, bop_rows, args.output_dir)

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {args.output_dir}")
    print("=" * 60)

    # Release vispy GL renderers to avoid segfault on exit.
    # Only relevant when need_bop=True; renderers are None otherwise.
    if need_bop:
        seen = set()
        for ds in datasets:
            r = ds.get("renderer")
            if r is not None and id(r) not in seen:
                seen.add(id(r))
                for method in ("close", "delete", "_close"):
                    fn = getattr(r, method, None)
                    if callable(fn):
                        try: fn()
                        except Exception: pass
                        break
    datasets.clear()

    import gc
    gc.collect()

if __name__ == "__main__":
    main()