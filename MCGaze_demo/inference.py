# inference.py
# Minimal-change inference wrapper for MCGaze.
# IMPORTANT: We intentionally keep the fragile "metas/config" setup exactly as in demo_real_time.py.
# Only image feeding is adjusted to actually use the provided clip frames.

from mmcv.parallel import collate, scatter

import cv2
import torch
import numpy as np
import sys
sys.path.insert(0, "..")
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose

# ====== BEGIN: directly copied config/init bits from demo_real_time.py (unchanged) ======
# use gaze360 or l2cs by changing these paths (kept verbatim)
model = init_detector(
        '../configs/multiclue_gaze/multiclue_gaze_r50_l2cs.py',
        '../ckpts/multiclue_gaze_r50_l2cs.pth',
        device="cuda:0",
        cfg_options=None,)

# model = init_detector(
#         '../configs/multiclue_gaze/multiclue_gaze_r50_gaze360.py',
#         '../ckpts/multiclue_gaze_r50_gaze360.pth',
#         device="cuda:0",
#         cfg_options=None,)

cfg = model.cfg
test_pipeline = Compose(cfg.data.test.pipeline[1:])

def load_datas(data, test_pipeline, datas):
    datas.append(test_pipeline(data))
# ====== END: copied init bits ======


def _to_unit(vec_np: np.ndarray) -> np.ndarray:
    """Normalize vectors along last dim; safe for zeros."""
    eps = 1e-8
    n = np.linalg.norm(vec_np, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return vec_np / n


def _build_clip_datas_from_frames(frames_bgr, approx_l: int = None):
    """
    Build the 'datas' list (per-frame items) using the SAME pipeline as demo_real_time.infer.
    frames_bgr: list[np.ndarray (H,W,3)] in BGR.
    """
    datas_list = []
    for frame in frames_bgr:
        h_n, w_n, _ = frame.shape  # (H,W,3)
        l = max(h_n, w_n) // 2 if approx_l is None else approx_l

        # Exactly as demo: filename=111, ori_filename=111, ori_shape=(2l,2l,3)
        cur_data = dict(
            filename=111,
            ori_filename=111,
            img=frame,  # BGR; Normalize(to_rgb=True) will handle channel swap
            img_shape=(h_n, w_n, 3),
            ori_shape=(2 * l, 2 * l, 3),
            img_fields=['img'],
        )
        load_datas(cur_data, test_pipeline, datas_list)

    return datas_list


def _forward_mcgaze_with_hacks(datas_list, clip_len):
    """
    EXACTLY mirrors the fragile meta/config manipulations in demo_real_time.infer,
    but (key change) we stack the actual per-frame tensors *before* collate.
    Steps:
      1) Extract each frame's post-pipeline tensor from datas_list (shape (3,H,W)).
      2) Stack to imgs_stack (T,3,H,W).
      3) Create a 'base' batch by collating ONLY the first item (samples_per_gpu=1).
      4) Overwrite img_metas (repeat the same dict clip_len times) and set img=[imgs_stack].
      5) Scatter to GPU and run the model.
    """
    # 1) Extract per-frame tensors from pipeline outputs *before* collate
    per_frame_tensors = []
    for sample in datas_list:
        img_obj = sample['img']
        img = img_obj.data if hasattr(img_obj, 'data') else img_obj
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if not torch.is_tensor(img):
            raise TypeError(f"Unexpected img type: {type(img)}")
        if img.dim() == 4 and img.size(0) == 1:
            img = img.squeeze(0)  # (1,3,H,W) -> (3,H,W)
        assert img.dim() == 3, f"Expected (3,H,W), got {tuple(img.shape)}"
        per_frame_tensors.append(img)

    # 2) Stack into (T,3,H,W)
    imgs_stack = torch.stack(per_frame_tensors, dim=0)

    # 3) Create a minimal 'base' batch by collating ONLY the first item
    base = collate([datas_list[0]], samples_per_gpu=1)
    base['img_metas'] = base['img_metas'].data
    base['img'] = base['img'].data

    # ====== BEGIN: fragile hack zone — keep constants *exactly* as in demo_real_time.py ======
    base["img_metas"] = [{
        'filename': 0,
        'ori_filename': 111,
        'ori_shape': (254, 254, 3),
        'img_shape': (448, 448, 3),
        'pad_shape': (448, 448, 3),
        'scale_factor': np.array([1.7637795, 1.7637795, 1.7637795, 1.7637795], dtype=float),
        'flip': False,
        'flip_direction': None,
        'img_norm_cfg': {
            'mean': np.array([123.675, 116.28 , 103.53 ], dtype=float),
            'std':  np.array([ 58.395,  57.12 ,  57.375], dtype=float),
            'to_rgb': True
        }
    }]
    base['img_metas'] = [[base['img_metas'][0] for _ in range(clip_len)]]

    # KEY CHANGE: use the real frames stack (T,3,H,W)
    base['img'] = [imgs_stack]
    # ====== END: fragile hack zone ======

    datas = scatter(base, ["cuda:0"])[0]

    # (Optional short sanity print; comment out if noisy)
    # print("imgs_stack shape:", tuple(imgs_stack.shape))
    # print("len(img_metas[0]):", len(base['img_metas'][0]))

    with torch.no_grad():
        (det_bboxes, det_labels), det_gazes = model(
            return_loss=False,
            rescale=True,
            format=False,
            **datas
        )

    gaze_dim = det_gazes['gaze_score'].size(1)
    det_fusion_gaze = det_gazes['gaze_score'].view((det_gazes['gaze_score'].shape[0], 1, gaze_dim))
    gaze_np = det_fusion_gaze.cpu().numpy()[:, 0, :]
    gaze_np = _to_unit(gaze_np)
    return gaze_np


def estimate_from_clip(clip, clip_size=None):
    """
    Run MCGaze on a full clip and return per-frame gaze vectors.

    Args:
        clip: list[np.ndarray] or np.ndarray of shape (T, H, W, 3), frames in BGR.
        clip_size: optional; if provided, we use min(len(clip), clip_size) frames (from the end).
                   If None, we use all frames.

    Returns:
        np.ndarray of shape (T_used, 3): unit gaze vectors per frame (model's output coordinate frame).
    """
    # Ensure list of np.ndarrays
    if isinstance(clip, np.ndarray):
        if clip.ndim != 4 or clip.shape[-1] != 3:
            raise ValueError(f"clip ndarray must be (T,H,W,3), got {clip.shape}")
        frames_bgr = [clip[i] for i in range(clip.shape[0])]
    elif isinstance(clip, list):
        frames_bgr = clip
    else:
        raise TypeError("clip must be a list of frames or a (T,H,W,3) ndarray")

    T = len(frames_bgr)
    if T == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Determine how many frames to use (local-only; no globals)
    use_T = T if clip_size is None else min(T, clip_size)
    frames_bgr_used = frames_bgr[-use_T:]

    # Build per-frame datas with SAME pipeline as demo_real_time.infer
    datas_list = _build_clip_datas_from_frames(frames_bgr_used)

    # Forward through MCGaze with the EXACT same meta/config hacks (only images differ)
    gaze_np = _forward_mcgaze_with_hacks(datas_list, clip_len=len(frames_bgr_used))

    return gaze_np


# ----------------- Tiny smoke test (optional) -----------------
if __name__ == "__main__":
    import os
    sample_path = "/home/kovan/FaceAndGaze/MCGaze_demo/1.png"  # put a real head crop here
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)  # BGR
        T = 5
        clip = [img.copy() for _ in range(T)]
        out = estimate_from_clip(clip)  # -> (T, 3)
        print("Output shape:", out.shape)
        print(out[:2])
    else:
        print("No real frame found for smoke test; skipping __main__ run.")
