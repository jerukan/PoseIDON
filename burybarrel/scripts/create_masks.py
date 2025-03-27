from pathlib import Path
from typing import List, Iterable

import click
import cv2
import dill as pickle
import numpy as np
from PIL import Image
from shapely import Polygon
from tqdm import tqdm

from burybarrel.image import imgs_from_dir
from burybarrel.langsam_utils import display_image_with_masks


@click.command()
@click.option(
    "-i",
    "--imgdir",
    "imgdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("-p", "--prompt", "text_prompt", required=True, type=click.STRING)
@click.option(
    "-o",
    "--outdir",
    "outdir",
    required=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "--boxthresh",
    "box_threshold",
    default=0.3,
    show_default=True,
    type=click.FLOAT,
    help="GroundingDINO option",
)
@click.option(
    "--textthresh",
    "text_threshold",
    default=0.25,
    show_default=True,
    type=click.FLOAT,
    help="GroundingDINO option",
)
@click.option(
    "--maskthresh",
    "mask_threshold",
    default=0.0,
    show_default=True,
    type=click.FLOAT,
    help="SAM option used to threshold logits into masks; lower if masks aren't including enough",
)
@click.option(
    "--closekernel",
    "closekernelsize",
    default=0,
    type=click.INT,
    show_default=True,
    help="n x n kernel size for morphological closing operation; set to 0 for no closing",
)
@click.option(
    "--convexhull",
    "convexhull",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Perform a convex hull on all masks if true",
)
@click.option(
    "-d",
    "--device",
    "device",
    type=click.STRING,
)
def create_masks(imgdir, text_prompt, outdir, box_threshold=0.3, text_threshold=0.25, mask_threshold=0.0, closekernelsize: int=0, convexhull=False, device=None):
    _create_masks(imgdir, text_prompt, outdir, box_threshold=box_threshold, text_threshold=text_threshold, mask_threshold=mask_threshold, closekernelsize=closekernelsize, convexhull=convexhull, device=device)

def _create_masks(imgdir, text_prompt, outdir, box_threshold=0.3, text_threshold=0.25, mask_threshold=0.0, closekernelsize: int=0, convexhull=False, device=None):
    from lang_sam import LangSAM
    from lang_sam.models import sam
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    imgdir = Path(imgdir)
    imgpaths, _ = imgs_from_dir(imgdir)
    outdir = Path(outdir)
    maskcomp_dir = outdir / "maskcomp"
    maskcomp_dir.mkdir(parents=True, exist_ok=True)
    maskdebug_dir = outdir / "maskdebug"
    maskdebug_dir.mkdir(parents=True, exist_ok=True)

    # langsam_model = LangSAM(sam_type="sam2.1_hiera_small")
    langsam_model = LangSAM(sam_type="sam2.1_hiera_large", device=device)
    # hook in SAM model with different parameters
    sam_model = sam.SAM()
    sam_model.build_model(langsam_model.sam_type, device=device)
    sam_model.predictor = SAM2ImagePredictor(sam_model.model, mask_threshold=mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0)
    langsam_model.sam = sam_model

    # each box is [x_min, y_min, x_max, y_max]
    bboxes = []
    for i, imgpath in enumerate(tqdm(imgpaths)):
        image_pil = Image.open(imgpath).convert("RGB")
        w, h = image_pil.size

        results = langsam_model.predict(
            [image_pil], [text_prompt], box_threshold=box_threshold, text_threshold=text_threshold
        )[0]
        boxes = results["boxes"]
        masks = results["masks"]
        scores = results["scores"]  # bbox score from dino
        mask_scores = np.array(results["mask_scores"])  # mask score from sam
        if len(mask_scores.shape) == 0:
            mask_scores = mask_scores[None, ...]

        if len(masks) == 0:
            print(f"No objects of the '{text_prompt}' prompt detected in image {imgpath}")
            continue
        masks_np = [(mask * 255).astype(np.uint8) for mask in masks]
        if closekernelsize > 0:
            # kernel = np.ones((closekernelsize, closekernelsize))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closekernelsize, closekernelsize))
            masks_np = [cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel) for mask_np in masks_np]
        if convexhull:
            contours = [cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for mask in masks_np]
            convexcontours = [cv2.convexHull(np.vstack(contour)) for contour in contours]
            masks_np = [cv2.drawContours(np.zeros_like(masks_np[0]), [convexcontour], -1, 255, thickness=cv2.FILLED) for convexcontour in convexcontours]

        bbox_mask_path = maskcomp_dir / f"{imgpath.stem}_img_with_mask.png"
        bbox_mask_path.parent.mkdir(parents=True, exist_ok=True)
        display_image_with_masks(image_pil, masks_np, boxes, scores, mask_scores, figwidth=13, savefig=bbox_mask_path, all_masks=True, show=False, show_confidence=True)

        # jank workaround for excluding those masks that are just supersets
        # of the barrel itself
        # reminder: [x_min, y_min, x_max, y_max]
        # as it turns out jellyfish are detected for some reason, and they are smaller
        # welp, just take the first one, it's usually the barrel anyway right?
        rects = np.array([Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]) for box in boxes])
        boxareas = np.array([rect.area for rect in rects])
        maskareas = np.array([np.sum(mask_np > 0) for mask_np in masks_np])
        validmask = np.ones(len(rects), dtype=bool)
        for i in range(len(rects)):
            if not validmask[i]:
                continue
            if rects[i].area > 0.8 * w * h:
                # specifically for this data, objects should not cover more than 80% of the
                # image (in fact, it probably should not cover more than 20%)
                # this is specifically for those goofy bounding boxes that cover the whole
                # image for some reason
                validmask[i] = False
                continue
            # for j in range(i+1, len(rects)):
            #     # remove superset bounding boxes
            #     if not validmask[j]:
            #         continue
            #     if rects[i].contains(rects[j]):
            #         validmask[i] = False
            #     elif rects[j].contains(rects[i]):
            #         validmask[j] = False
        # best_idx = np.argmin(boxareas)
        if len(maskareas[validmask]) == 0:
            print(f"No valid masks after filtering for {imgpath}")
            continue
        best_idx = np.arange(len(boxes))[validmask][np.argmax(scores[validmask])]

        # save masks
        for i, mask_np in enumerate(masks_np):
            # each box is x_min, y_min, x_max, y_max
            bbox = boxes[i]
            mask_path = maskdebug_dir / f"{imgpath.stem}_mask_{i+1}.png"
            mask_image = Image.fromarray(mask_np)
            mask_image.save(mask_path)

        bbox = boxes[best_idx]
        mask_np = masks_np[best_idx]
        mask_path = outdir / f"{imgpath.stem}.png"
        mask_image = Image.fromarray(mask_np)
        mask_image.save(mask_path)
        bboxes.append(bbox)

    bboxes = np.array(bboxes, dtype=int)
    with open(outdir / "bboxes.pickle", "wb") as f:
        pickle.dump(bboxes, f)
