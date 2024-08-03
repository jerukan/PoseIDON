"""
Utility module for grabbing screenshots from ROV footage.

For new dives, you must manually add the video paths and the start
timestamps for each video.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# the order of videos inside the video directory for a dive
# hardcoded paths due to potential of ambiguous order of paths
# the second element is the start for each video in each dive (PST)
# they have to be hardcoded due to possibility of missing footage
# add to this dict as we get more dives
# relative to the parent directory (vid_dir)
divenum_to_info = {
    1: [
        ["Capture0000.mov", "2023-04-05 14:41:56"],
        ["Capture0001.mov", "2023-04-05 17:14:19"],
        ["Capture0002.mov", "2023-04-05 19:45:35"],
        ["Capture0003.mov", "2023-04-05 19:58:39"],
        ["Capture0004.mov", "2023-04-05 22:31:02"],
        ["Capture0005.mov", "2023-04-06 01:03:25"],
        ["Capture0006.mov", "2023-04-06 03:35:48"],
    ],
    2: [
        ["Capture0005.mov", "2023-04-06 14:59:49"],
    ],
    3: [
        ["Capture0000.mov", "2023-04-07 02:18:23"],
        ["Capture0001.mov", "2023-04-07 04:50:46"],
        ["Capture0002.mov", "2023-04-07 07:23:09"],
        ["Capture0003.mov", "2023-04-07 09:55:32"],
        ["Capture0004.mov", "2023-04-07 12:27:55"],
        ["Capture0005.mov", "2023-04-07 15:00:18"],
        ["Capture0006.mov", "2023-04-07 17:32:41"],
        ["Capture0007.mov", "2023-04-07 20:05:04"],
        ["Capture0008.mov", "2023-04-07 22:37:28"],
        ["Second Disk/Capture0000.mov", "2023-04-08 00:54:51"],
        ["Second Disk/Capture0001.mov", "2023-04-08 03:27:14"],
        ["Second Disk/Capture0002.mov", "2023-04-08 05:59:37"],
        ["Second Disk/Capture0003.mov", "2023-04-08 08:32:00"],
        ["Second Disk/Capture0004.mov", "2023-04-08 11:04:22"],
        ["Second Disk/Capture0005.mov", "2023-04-08 13:36:45"],
    ],
    4: [
        ["Capture0000.mov", "2023-04-09 05:20:46"],
        ["Capture0001.mov", "2023-04-09 07:53:09"],
        ["Capture0002.mov", "2023-04-09 10:25:32"],
        ["Capture0003.mov", "2023-04-09 12:57:55"],
        ["Capture0004.mov", "2023-04-09 15:30:18"],
        ["Capture0005.mov", "2023-04-09 18:02:41"],
        ["Capture0006.mov", "2023-04-09 20:35:05"],
        ["Capture0007.mov", "2023-04-09 23:07:28"],
        ["Capture0008.mov", "2023-04-10 01:39:51"],
        ["Capture0009.mov", "2023-04-10 04:12:14"],
        ["Capture0010.mov", "2023-04-10 06:44:37"],
    ],
    5: [
        ["Capture0000.mov", "2023-04-10 19:23:20"],
        ["Capture0001.mov", "2023-04-10 21:55:43"],
    ],
    6: [
        ["Capture0000.mov", "2023-04-13 07:19:23"],
        ["Capture0001.mov", "2023-04-13 09:51:46"],
        ["Capture0002.mov", "2023-04-13 12:24:09"],
        ["Capture0003.mov", "2023-04-13 14:56:32"],
    ],
    7: [
        ["Capture0000.mov", "2023-04-13 19:53:25"],
        ["Capture0001.mov", "2023-04-13 22:25:49"],
        ["Capture0002.mov", "2023-04-14 00:58:12"],
        ["Capture0003.mov", "2023-04-14 03:30:35"],
    ],
    8: [
        ["Capture0000.mov", "2023-04-14 09:17:27"],
        ["Capture0001.mov", "2023-04-14 11:49:50"],
        ["Capture0002.mov", "2023-04-14 14:22:13"],
        ["Capture0003.mov", "2023-04-14 16:54:36"],
    ],
    9: [
        ["Capture0000.mov", "2023-04-15 02:32:41"],
        ["Capture0001.mov", "2023-04-15 05:05:04"],
        ["Capture0002.mov", "2023-04-15 07:37:27"],
    ],
    10: [
        ["Capture0000.mov", "2023-04-15 16:00:58"],
        ["Capture0001.mov", "2023-04-15 18:33:21"],
        ["Capture0002.mov", "2023-04-15 21:05:44"],
        ["Capture0003.mov", "2023-04-15 23:38:07"],
        ["Capture0004.mov", "2023-04-16 02:10:30"],
        ["Capture0005.mov", "2023-04-16 04:42:53"],
        ["Capture0006.mov", "2023-04-16 07:15:16"],
        ["Capture0007.mov", "2023-04-16 09:47:39"],
        ["Capture0008.mov", "2023-04-16 12:20:02"],
        ["Capture0009.mov", "2023-04-16 14:52:25"],
        ["Capture0010.mov", "2023-04-16 17:24:48"],
        ["Capture0011.mov", "2023-04-16 19:57:11"],
        ["Capture0012.mov", "2023-04-16 22:02:14"],
        ["Capture0013.mov", "2023-04-17 00:34:37"],
        ["Capture0014.mov", "2023-04-17 03:07:00"],
        ["Capture0015.mov", "2023-04-17 05:39:23"],
    ],
    11: [
        ["Capture0000.mov", "2023-04-17 17:21:52"],
        ["Capture0001.mov", "2023-04-17 19:54:15"],
    ],
    12: [
        ["Capture0000.mov", "2023-04-18 04:41:34"],
        ["Capture0001.mov", "2023-04-18 07:13:57"],
        ["Capture0002.mov", "2023-04-18 09:46:20"],
        ["Capture0003.mov", "2023-04-18 12:18:43"],
        ["Capture0004.mov", "2023-04-18 14:51:06"],
        ["Capture0005.mov", "2023-04-18 17:23:29"],
    ],
    13: [
        ["Capture0000.mov", "2023-04-19 04:29:00"],
        ["Capture0001.mov", "2023-04-19 07:01:23"],
        ["Capture0002.mov", "2023-04-19 09:33:46"],
        ["Capture0003.mov", "2023-04-19 12:06:09"],
        ["Capture0004.mov", "2023-04-19 14:38:32"],
    ],
    14: [
        ["Capture0000.mov", "2023-04-20 03:19:54"],
        ["Capture0001.mov", "2023-04-20 05:52:17"],
        ["Capture0002.mov", "2023-04-20 08:24:40"],
    ],
    15: [
        ["Capture0000.mov", "2023-04-20 16:56:51"],
    ],
    17: [
        ["Capture0000.mov", "2023-04-22 17:48:09"],
    ],
    18: [
        ["Capture0000.mov", "2023-04-23 03:15:28"],
        ["Capture0001.mov", "2023-04-23 05:47:51"],
        ["Capture0002.mov", "2023-04-23 08:20:14"],
        ["Capture0003.mov", "2023-04-23 10:52:37"],
        ["Capture0004.mov", "2023-04-23 13:25:00"],
        ["Capture0005.mov", "2023-04-23 15:57:23"],
        ["Capture0006.mov", "2023-04-23 18:29:46"],
        ["Capture0007.mov", "2023-04-23 21:02:09"],
    ],
}

# DON'T modify these
divenum = None
vid_dir = None
vidpaths = None

for key, val in divenum_to_info.items():
    for i, (_, ts) in enumerate(val):
        val[i][1] = pd.Timestamp(ts).tz_localize("America/Los_Angeles")
    divenum_to_info[key] = np.array(val)
vidstarts = None
vids = None


def set_dive_info(divenum_, vid_dir_=None):
    """
    Use this to update the dive info in this file.

    Args:
        divenum_ (int)
        vid_dir_ (path-like): Defaults to the path on the NAS.
    """
    global divenum, vid_dir
    divenum = divenum_
    if vid_dir_ is None:
        vid_dir_ = f"/Volumes/DDT_23/data/ROV/CURV Video/Dive {divenum}/HD Camera/"
    vid_dir = Path(vid_dir_)
    if not vid_dir.exists():
        raise FileNotFoundError(f"{vid_dir} not found. Are you connected to the NAS?")
    populate_vids()


def populate_vids():
    global vidpaths, vidstarts, vids
    vidpaths = [vid_dir / path for path in divenum_to_info[divenum][:, 0]]
    vidstarts = divenum_to_info[divenum][:, 1]
    vids = [cv2.VideoCapture(str(path)) for path in vidpaths]


def release_vids():
    """This is problematic, don't use this."""
    for vid in vids:
        vid.release()


def get_frame(target_timestamp, ignore_oob=False, verbose=False, outdir=None):
    """
    Saves a frame from the total ROV footage given a local timestamp.

    Args:
        target_timestamp (pd.Timestamp): Must be timezone-aware.
        ignore_oob (bool): Set to true to silence exceptions from a timestamp
            existing in the video footage.
        verbose (bool)
        outdir (path-like): Directory to save the frame captures to.

    Returns:
        path-like: The path to a picture of the captured frame.
    """
    if divenum is None:
        raise ValueError("Run set_dive_info first!")
    if vidstarts is None:
        return None
    target_timestamp = pd.Timestamp(target_timestamp)
    vididx = None
    vidoffset = None
    for i, (vid, vidstart) in enumerate(zip(vids, vidstarts)):
        if i < len(vidpaths) - 1:
            next_start = vidstarts[i + 1]
            if vidstart <= target_timestamp and target_timestamp < next_start:
                vididx = i
                vidoffset = (target_timestamp - vidstart) / np.timedelta64(1, "s")
                break
        else:
            vididx = i
            vidoffset = (target_timestamp - vidstart) / np.timedelta64(1, "s")

    vid = vids[vididx]
    nframes = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    framenum = int(fps * vidoffset)
    if framenum > nframes:
        if ignore_oob:
            return None
        raise ValueError(
            f"Exceeded maximum number of frames for video {vidpaths[vididx]}. Your timestamp ({target_timestamp}) is either out of range or is missing from the data itself!"
        )
    vid.set(cv2.CAP_PROP_POS_FRAMES, framenum)
    _, frame = vid.read()
    if outdir is None:
        outdir = Path(f"ROV_Dive{str(divenum).zfill(2)}", "Images")
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    outpath = (
        outdir
        / f"rov-dive{divenum}-capture-{target_timestamp.strftime('%Y-%m-%dT%H-%M-%S')}.png"
    )
    cv2.imwrite(str(outpath), frame)
    if verbose:
        print(f"Wrote video frame to {outpath}")
    return outpath
