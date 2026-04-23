import os
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from gaze_estimation import GazeModel
from tqdm import tqdm
import cv2 as cv

# ---- Gaze3D (GaT) pieces copied from demo.py -------------------------
from src.models.gat_model import GaT, HeadDict, MLPHead, Swin3D


def _batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield i, iterable[i : i + batch_size]

class Gaze3D(GazeModel):
    """
    Estimate gaze from RGB head crops using the GaT (Swin3D) model from demo.py.
    Mirrors the 'estimate_from_crops' contract of PureGaze:
      - Inputs: list of HxWx3 RGB crops (np.uint8/float-like), timestamps (float array)
      - Output: (gaze_with_ts, valid_indices)
    Notes:
      * This implements the 'image' modality (T=1); for video clips supply a list of
        per-crop clips and set 'clip_len' accordingly.
    """
    def __init__(
        self,
        ckpt_path = os.getcwd()+"/checkpoints/gat_gaze360.ckpt",
        device: str = "cuda:0",
        clip_len: int = 1,             # 1 == image modality
        stride: int = 1,
        batch_size: int = 16,          # match demo defaults where possible
        resize_hw: Tuple[int, int] = (224, 224),  # reasonable default
        rectification = False,          # rectification setting,
        isRGB = False,
        save_input_frames_for_debug: bool = False,  # if True, saves some input frames to ./gaze3d_debug_frames for debugging
    ):
        super().__init__(rectification=rectification, isRGB=isRGB)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.batch_size = int(batch_size)
        self.resize_hw = resize_hw
        self._full_gaze_tensor = None  # will be filled per-call in estimate_from_crops

        # Build GaT as in demo.py
        self.model = GaT(
            encoder=Swin3D(pretrained=False),
            head_dict=HeadDict(
                names=["gaze"],
                modules=[partial(MLPHead, hidden_dim=256, num_layers=1, out_features=3)],
            ),
        )

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"GaT checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        # demo.py loads from ["state_dict"] and uses strict=True
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Simple preprocessing (PIL -> resize -> to tensor). Demo’s transforms live in DemoImageData;
        # we mimic a minimal, safe pipeline here.
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_hw),
            transforms.ToTensor(),  # [0,1], CHW
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ) #ImageNet Normalization for the ResNet backbone
        ])
    
        self.save_input_frames_for_debug = save_input_frames_for_debug  # set to True to save some input frames for debugging

    def get_model_name(self) -> str:
        if self.rectification:
            return "gaze3d_clip_len_" + str(self.clip_len) + "_rectification"
        return "gaze3d_clip_len_" + str(self.clip_len)
    
    @staticmethod
    def _sliding_windows(N: int, T: int, stride: int = 1):
        if N < T:
            return
        i = 0
        while i + T <= N:
            start = i
            end = i + T
            yield start, end, end - 1  # (start, end, last-index)
            i += stride


    def estimate_from_crops(
        self,
        head_crops: list[np.ndarray],  # sequence for ONE track: N x H x W x 3 (RGB)
        timestamps: np.ndarray,        # shape [N]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
        gaze_with_ts: float64 [M, 4] -> [timestamp(mid), gx, gy, gz]
        valid_indices: int32  [M] -> index of the CENTER frame each prediction corresponds to
        Behavior mirrors demo.py:
        - L2 normalize gaze
        - choose middle t (for even T: t = T//2 - 1)
        """
        N = len(head_crops)
        if N == 0:
            return np.empty((0, 4), np.float64), np.empty((0,), np.int32)
        if timestamps.shape[0] != N:
            raise ValueError("head_crops and timestamps must align")

        T = int(self.clip_len)
        if T <= 0:
            raise ValueError("clip_len must be >= 1")

        # Prepare full tensor: right-aligned rows per t (here rows are just length T)
        full_tensor = np.zeros((N, T, 3), dtype=np.float64)

        rows = []      # [timestamp[t], gx, gy, gz]
        centers = []   # t indices (last index of each window)

        win_ranges = list(self._sliding_windows(N, T, self.stride))
        if len(win_ranges) == 0:
            self._full_gaze_tensor = full_tensor
            return np.empty((0, 4), np.float64), np.empty((0,), np.int32)

        batch_clips = []
        batch_t_last = []

        def _flush():
            nonlocal batch_clips, batch_t_last, rows, centers, full_tensor
            if not batch_clips:
                return
            X = torch.stack(batch_clips, dim=0).to(self.device)  # (B, T, 3, H, W)
            if self.save_input_frames_for_debug:
                self._save_model_input_frame(X, batch_t_last[0], save_dir="./gaze3d_debug_frames", max_images=12)  # <<< add this
            with torch.no_grad():
                out = self.model(X)                               # dict: "gaze" -> (B, T, 3)
                g = F.normalize(out["gaze"], p=2, dim=2, eps=1e-8).detach().cpu().numpy()  # (B, T, 3)

            # Fill full_tensor rows and extract THE VECTOR = last frame's vector
            for bi, t_last in enumerate(batch_t_last):
                # transform all T vectors for this window
                g_all = g[bi]  # (T, 3)
                transformed = np.empty((T, 3), dtype=np.float64)
                for j in range(T):
                    transformed[j, :] = self.transform_gaze_to_custom_basis(g_all[j, :])

                # save full row for this t (already length T)
                full_tensor[t_last, :, :] = transformed

                # pick last frame's vector as THE VECTOR
                v_last = transformed[-1, :]
                rows.append([float(timestamps[t_last]), float(v_last[0]), float(v_last[1]), float(v_last[2])])
                centers.append(int(t_last))

            batch_clips.clear()
            batch_t_last.clear()

        # Build batches from causal windows
        for (start, end, t_last) in tqdm(win_ranges, desc="Gaze3D (causal) estimation"):
            clip_imgs = []
            for k in range(start, end):
                img = head_crops[k]
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                clip_imgs.append(self.preprocess(img))  # (3,H,W)
            batch_clips.append(torch.stack(clip_imgs, dim=0))  # (T,3,H,W)
            batch_t_last.append(t_last)

            if len(batch_clips) == self.batch_size:
                _flush()

        # flush remainder
        _flush()

        self._full_gaze_tensor = full_tensor

        if len(rows) == 0:
            return np.empty((0, 4), np.float64), np.empty((0,), np.int32)

        gaze_with_ts = np.asarray(rows, dtype=np.float64)
        valid_indices = np.asarray(centers, dtype=np.int32)
        return gaze_with_ts, valid_indices


    def save_estimation(self, gaze_directions: np.ndarray, valid_indices: np.ndarray,
                        output_dir: str, append_model_name_to_output_path: bool = True):
        # save the usual files
        super().save_estimation(gaze_directions, valid_indices, output_dir, append_model_name_to_output_path)

        base_output_path = output_dir
        if append_model_name_to_output_path:
            base_output_path = os.path.join(base_output_path, self.get_model_name())
        os.makedirs(base_output_path, exist_ok=True)

        if getattr(self, "_full_gaze_tensor", None) is not None:
            np.save(os.path.join(base_output_path, "gaze_directions_full.npy"), self._full_gaze_tensor)


    @staticmethod
    def transform_gaze_to_custom_basis(vec: np.ndarray) -> np.ndarray:
        vec = vec / np.linalg.norm(vec)
        return np.array([vec[2], vec[0], vec[1]])  # [z, x, y]

    def _save_model_input_frame(self, X: torch.Tensor, t_last: int, save_dir: str, max_images: int = 12):
        """
        Save the last frame (the one used for the causal prediction) from the first sample in X.
        - X: (B, T, 3, H, W) float32 in [0,1] on device
        - t_last: int index of the last frame for this window (for filename context)
        - save_dir: where to write PNGs
        - max_images: stop after saving this many images (persists across calls)
        """
        # A tiny persistent counter; attach to self the first time.
        if not hasattr(self, "_viz_saved"):
            self._viz_saved = 0
        if self._viz_saved >= max_images:
            return

        os.makedirs(save_dir, exist_ok=True)

        # First sample, last frame in its window
        x0_last = X[0, -1]                      # (3, H, W), float in [0,1]
        if x0_last.is_cuda:
            x0_last = x0_last.detach().cpu()
        img_rgb = (x0_last.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)  # (H, W, 3) RGB
        img_bgr = img_rgb[..., ::-1]            # OpenCV expects BGR

        out_path = os.path.join(save_dir, f"model_input_{self._viz_saved:04d}_t{int(t_last)}.png")
        cv.imwrite(out_path, img_bgr)
        self._viz_saved += 1
