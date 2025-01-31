import cv2
import numpy as np
import visu3d as v3d


def segment_pc_from_mask(pc: v3d.Point3d, mask, v3dcam: v3d.Camera):
    idxs = np.arange(pc.shape[0])
    H, W = v3dcam.spec.resolution
    pxpts = v3dcam.px_from_world @ pc
    uvs = pxpts.p
    valid = (uvs[:, 0] >= 0) & (uvs[:, 0] <= W) & (uvs[:, 1] >= 0) & (uvs[:, 1] <= H)
    barrelmask = mask[uvs[valid].astype(int).T[1], uvs[valid].astype(int).T[0]] > 0
    barrelidxs = idxs[valid][barrelmask]
    return barrelidxs


def get_bbox_mask(bbox, W, H):
    """
    Sets values inside bounding box to 255.

    Args:
        bbox: [x_min, y_min, x_max, y_max]
    """
    bbox = np.array(bbox, dtype=int)
    boxmask = np.zeros((H, W), dtype=np.uint8)
    boxmask = cv2.rectangle(boxmask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
    return boxmask


def get_local_plane_mask(bbox, expandratio_in, expandratio_out, W, H):
    """
    Takes the difference between a larger bbox and an even larger bbox mask to get
    a 'frame' of the local plane around the barrel.

    Args:
        bbox: [x_min, y_min, x_max, y_max]
        expandratio_in: expansion ratio of bbox sides for inner bbox
        expandratio_out: expansion ratio of bbox sides for outer bbox
    """
    newbboxout = np.zeros(4, dtype=int)
    newbboxin = np.zeros(4, dtype=int)
    expandratioin = expandratio_in
    expandratioout = expandratio_out
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    cx = bbox[0] + width // 2
    cy = bbox[1] + height // 2
    newbboxin[0] = max(cx - (expandratioin * width) // 2, 0)
    newbboxin[1] = max(cy - (expandratioin * height) // 2, 0)
    newbboxin[2] = min(cx + (expandratioin * width) // 2, W)
    newbboxin[3] = min(cy + (expandratioin * height) // 2, H)
    newbboxout[0] = max(cx - (expandratioout * width) // 2, 0)
    newbboxout[1] = max(cy - (expandratioout * height) // 2, 0)
    newbboxout[2] = min(cx + (expandratioout * width) // 2, W)
    newbboxout[3] = min(cy + (expandratioout * height) // 2, H)
    return get_bbox_mask(newbboxout, W, H) - get_bbox_mask(newbboxin, W, H)


def apply_clahe(img, clipLimit=None, tileGridSize=None):
    """
    Applies CLAHE to the image in CIELAB colorspace, as specified in the following link

    https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def render_v3d(cam: v3d.Camera, points: v3d.Point3d, radius=1) -> np.ndarray:
    """
    A modified version of v3d.Camera.render() to allow the sizes of each of the
    points in a point cloud to be changed, since you literally couldn't see
    anything for sparse point clouds renders (each points is literally a pixel).
    Currently doesn't scale point size for distance. I may implement this later.

    Project 3d points to the camera screen.

    Args:
      points: 3d points.

    Returns:
      img: The projected 3d points.
    """
    # TODO(epot): Support float colors and make this differentiable!
    if not isinstance(points, v3d.Point3d):
        raise TypeError(
            f'Camera.render expect `v3d.Point3d` as input. Got: {points}.'
        )

    # Project 3d -> 2d coordinates
    points2d = cam.px_from_world @ points

    # Flatten pixels
    points2d = points2d.flatten()
    px_coords = points2d.p
    rgb = points2d.rgb

    # Compute the valid coordinates
    w_coords = px_coords[..., 0]
    h_coords = px_coords[..., 1]
    valid_coords_mask = (
        (0 <= h_coords)
        & (h_coords < cam.h - 1)
        & (0 <= w_coords)
        & (w_coords < cam.w - 1)
        & (points2d.depth[..., 0] > 0)  # Filter points behind the camera
    )
    rgb = rgb[valid_coords_mask]
    px_coords = px_coords[valid_coords_mask]
    px_coords = np.astype(np.round(px_coords), np.int32)

    # px_coords is (h, w)
    img = np.zeros((*cam.resolution, 3), dtype=np.uint8)
    for i, coord in enumerate(px_coords):
        img = cv2.circle(img, coord, radius, tuple(int(ch) for ch in rgb[i]), -1)
    return img
