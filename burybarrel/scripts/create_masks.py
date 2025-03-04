from pathlib import Path
from typing import List, Iterable

import click
import cv2
import dill as pickle
import numpy as np
from PIL import Image
from shapely import Polygon
from tqdm import tqdm

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
)
@click.option(
    "--textthresh",
    "text_threshold",
    default=0.25,
    show_default=True,
    type=click.FLOAT,
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
    help="Perform a convex hull on all masks if true (closing is ignored if this is true)",
)
def create_masks(imgdir, text_prompt, outdir, box_threshold=0.3, text_threshold=0.25, closekernelsize: int=0, convexhull=False):
    from lang_sam import LangSAM
    from lang_sam.models.sam import SAM
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    imgdir = Path(imgdir)
    imgpaths = sorted(list(imgdir.glob("*.png")) + list(imgdir.glob("*.jpg")))
    outdir = Path(outdir)
    maskcomp_dir = outdir / "maskcomp"
    maskcomp_dir.mkdir(parents=True, exist_ok=True)
    maskdebug_dir = outdir / "maskdebug"
    maskdebug_dir.mkdir(parents=True, exist_ok=True)

    # langsam_model = LangSAM(sam_type="sam2.1_hiera_small")
    langsam_model = LangSAM(sam_type="sam2.1_hiera_large")
    # hook in SAM model with different parameters
    sam_model = SAM()
    sam_model.build_model(langsam_model.sam_type)
    sam_model.predictor = SAM2ImagePredictor(sam_model.model, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
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
        logits = results["scores"]

        if len(masks) == 0:
            print(f"No objects of the '{text_prompt}' prompt detected in the image.")
        else:
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
            display_image_with_masks(image_pil, masks_np, boxes, logits, figwidth=13, savefig=bbox_mask_path, all_masks=True, show=False, show_confidence=True)

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
            maxvalididx = np.argmax(maskareas[validmask])
            best_idx = np.arange(len(rects))[validmask][maxvalididx]

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
