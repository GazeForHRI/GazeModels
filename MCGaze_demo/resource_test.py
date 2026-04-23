#!/usr/bin/env python3
# resource_test.py (MCGaze, standalone; Gaze3D-parity outputs)
import json
import time
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- ensure we import the project's compatible mmdet (like inference.py) ---
import sys
sys.path.insert(0, "..")

from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose


# ============================= FLOPs helpers (no adapter) =============================

def _has_fvcore() -> bool:
    try:
        import fvcore  # noqa: F401
        return True
    except Exception:
        return False


def _has_thop() -> bool:
    try:
        import thop  # noqa: F401
        return True
    except Exception:
        return False


def _minimal_img_meta(H_src: int, W_src: int) -> Dict[str, Any]:
    """
    Minimal meta dict mirroring your fragile block.
    Values won’t affect FLOPs; keys/shapes must match.
    """
    return {
        'filename': 0,
        'ori_filename': 111,
        'ori_shape': (H_src, W_src, 3),
        'img_shape': (448, 448, 3),
        'pad_shape': (448, 448, 3),
        'scale_factor': np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        'flip': False,
        'flip_direction': None,
        'img_norm_cfg': {
            'mean': np.array([123.675, 116.28, 103.53], dtype=float),
            'std':  np.array([58.395, 57.12, 57.375], dtype=float),
            'to_rgb': True
        }
    }


def _compute_fvcore_flops_per_clip(model: nn.Module,
                                   device: torch.device,
                                   T: int, C: int, H: int, W: int) -> Optional[int]:
    """Monkey-patch model.forward(x:[T,C,H,W]) -> Tensor(gaze_score) and run fvcore on it."""
    from fvcore.nn import FlopCountAnalysis
    sample_TCHW = torch.randn(T, C, H, W, device=device, dtype=torch.float32)
    meta = _minimal_img_meta(H, W)

    model.eval()
    orig_forward = model.forward
    try:
        def proxy_forward(x, *args, **kwargs):
            img = [x]  # list length 1
            img_metas = [[meta for _ in range(T)]]
            ((_, _), det_gazes) = orig_forward(
                return_loss=False, rescale=True, format=False,
                img=img, img_metas=img_metas
            )
            return det_gazes["gaze_score"]

        model.forward = proxy_forward
        with torch.no_grad():
            fca = FlopCountAnalysis(model, (sample_TCHW,))
            flops = int(fca.total())
        return flops if flops > 0 else None
    finally:
        model.forward = orig_forward


def _compute_thop_macs_per_clip(model: nn.Module,
                                device: torch.device,
                                T: int, C: int, H: int, W: int) -> Optional[int]:
    """
    Run THOP on a *deep-copied* model so hooks never affect the live model.
    """
    from thop import profile
    import copy

    sample_TCHW = torch.randn(T, C, H, W, device=device, dtype=torch.float32)
    meta = _minimal_img_meta(H, W)

    m = copy.deepcopy(model).to(device).eval()
    orig_forward = m.forward
    try:
        def proxy_forward(x, *args, **kwargs):
            img = [x]
            img_metas = [[meta for _ in range(T)]]
            ((_, _), det_gazes) = orig_forward(
                return_loss=False, rescale=True, format=False,
                img=img, img_metas=img_metas
            )
            return det_gazes["gaze_score"]

        m.forward = proxy_forward
        with torch.no_grad():
            macs, _ = profile(m, inputs=(sample_TCHW,), verbose=False)
        return int(macs) if macs and macs > 0 else None
    finally:
        m.forward = orig_forward


def _thop_artifact_cleanup(model: nn.Module) -> None:
    """If THOP ever touched the live model, strip leftover attrs."""
    for mod in model.modules():
        for attr in ("total_ops", "total_params"):
            if hasattr(mod, attr):
                try:
                    delattr(mod, attr)
                except Exception:
                    pass


# ============================= MCGaze resource tester =============================

class MCGazeResourceTester:
    """
    Standalone resource tester for MCGaze that:
      - builds test_pipeline from cfg
      - processes frames through Compose exactly like your inference.py
      - stacks real clip tensors before collate
      - overwrites img_metas/img exactly like your hack block
      - measures latency/VRAM per frame (like resource_section_for_mcgaze)
      - computes FLOPs per forward-clip (fvcore → thop fallback)
    """

    def __init__(self, config_path: str, checkpoint_path: str, device_str: str, clip_size: int):
        self.model = init_detector(config_path, checkpoint_path, device=device_str, cfg_options=None).eval()
        self.cfg = self.model.cfg
        self.test_pipeline = Compose(self.cfg.data.test.pipeline[1:])
        self.device = torch.device(device_str)
        self.clip_size = int(clip_size)
        torch.backends.cudnn.benchmark = True  # fixed shapes => faster kernels

    # ---------- pipeline helpers (mirror inference.py) ----------

    def _build_datas_for_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Run test_pipeline on a single raw BGR frame (uint8 HxWx3), like inference.py."""
        h_n, w_n, _ = frame_bgr.shape
        l = max(h_n, w_n) // 2
        cur_data = dict(
            filename=111,
            ori_filename=111,
            img=frame_bgr,   # BGR; pipeline has to_rgb=True
            img_shape=(h_n, w_n, 3),
            ori_shape=(2 * l, 2 * l, 3),
            img_fields=['img'],
        )
        return self.test_pipeline(cur_data)

    def _collate_and_stack_clip(self, datas_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Tuple[int, int, int, int]]:
        """Extract per-frame tensors -> stack to (T,3,H,W) -> collate base -> overwrite metas/img -> scatter."""
        # 1) stack post-pipeline TCHW
        per_frame_tensors: List[torch.Tensor] = []
        for sample in datas_list:
            img_obj = sample['img']
            img = img_obj.data if hasattr(img_obj, 'data') else img_obj
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dim() == 4 and img.size(0) == 1:
                img = img.squeeze(0)  # (1,3,H,W) -> (3,H,W)
            assert img.dim() == 3, f"Expected (3,H,W), got {tuple(img.shape)}"
            per_frame_tensors.append(img)
        imgs_stack = torch.stack(per_frame_tensors, dim=0)  # [T,3,H,W]
        T, C, H, W = [int(x) for x in imgs_stack.shape]

        # 2) collate only the first item
        base = collate([datas_list[0]], samples_per_gpu=1)
        base['img_metas'] = base['img_metas'].data
        base['img'] = base['img'].data

        # 3) overwrite metas/img exactly like the fragile block
        base["img_metas"] = [[
            {
                'filename': 0,
                'ori_filename': 111,
                'ori_shape': (H, W, 3),
                'img_shape': (448, 448, 3),
                'pad_shape': (448, 448, 3),
                'scale_factor': np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
                'flip': False,
                'flip_direction': None,
                'img_norm_cfg': {
                    'mean': np.array([123.675, 116.28, 103.53], dtype=float),
                    'std':  np.array([58.395, 57.12, 57.375], dtype=float),
                    'to_rgb': True
                }
            } for _ in range(T)
        ]]
        base['img'] = [imgs_stack]

        # 4) scatter to model device
        datas = scatter(base, [str(self.device)])[0]
        return datas, (T, C, H, W)

    # ---------- metrics ----------

    def _count_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return int(total), int(trainable)

    def _try_get_flops_thop(self, T: int, C: int, H: int, W: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns (thop_macs, thop_flops=2*macs). Deep-copied profiling to avoid hooks on live model.
        """
        if not _has_thop():
            return None, None
        try:
            macs = _compute_thop_macs_per_clip(self.model, self.device, T, C, H, W)
            if macs and macs > 0:
                return int(macs), int(2 * macs)
        except Exception as e:
            print(f"[THOP] failed: {e}")
        return None, None

    def _try_get_flops_fvcore(self, T: int, C: int, H: int, W: int) -> Optional[int]:
        if not _has_fvcore():
            return None
        try:
            fl = _compute_fvcore_flops_per_clip(self.model, self.device, T, C, H, W)
            if fl and fl > 0:
                return int(fl)
        except Exception as e:
            print(f"[fvcore] failed: {e}")
        return None

    def _time_one_forward(self, datas: Dict[str, Any], use_cuda: bool) -> float:
        """Per-frame latency for the clip forward (your real-time step)."""
        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device)
            starter.record()
            _ = self.model(return_loss=False, rescale=True, format=False, **datas)
            ender.record()
            torch.cuda.synchronize(self.device)
            return float(starter.elapsed_time(ender))  # ms
        else:
            t0 = time.perf_counter()
            _ = self.model(return_loss=False, rescale=True, format=False, **datas)
            t1 = time.perf_counter()
            return (t1 - t0) * 1000.0

    # ---------- main entry ----------

    def run(self, frames_bgr: np.ndarray, trials: Optional[int] = None) -> Dict[str, Any]:
        """
        frames_bgr: np.uint8 array [N,H,W,3] (BGR raw crops).
        trials: limit number of frames (default uses all).
        """
        assert isinstance(frames_bgr, np.ndarray) and frames_bgr.ndim == 4 and frames_bgr.shape[-1] == 3, \
            "frames_bgr must be uint8 numpy array with shape [N,H,W,3]"
        N_total = frames_bgr.shape[0]
        N = min(N_total, int(trials)) if trials is not None else N_total

        use_cuda = (self.device.type == "cuda" and torch.cuda.is_available())
        self.model.eval()
        torch.set_grad_enabled(False)

        # warm-up
        buffer: deque = deque(maxlen=self.clip_size)
        if use_cuda:
            dummy = np.zeros_like(frames_bgr[0])
            buffer.append(self._build_datas_for_frame(dummy))
            datas_warm, _ = self._collate_and_stack_clip(list(buffer) * self.clip_size)
            _ = self.model(return_loss=False, rescale=True, format=False, **datas_warm)
            torch.cuda.synchronize(self.device)
            buffer.clear()

        # params
        total_params, trainable_params = self._count_params()

        # Determine pipeline output shape using a real clip (repeat last if needed)
        first_items: List[Dict[str, Any]] = []
        need = self.clip_size
        idx = 0
        while len(first_items) < need and idx < N_total:
            first_items.append(self._build_datas_for_frame(frames_bgr[idx]))
            idx += 1
        if len(first_items) < need:
            last = first_items[-1]
            first_items += [last] * (need - len(first_items))
        _, (T, C, H, W) = self._collate_and_stack_clip(first_items)

        # FLOPs / MACs
        fvcore_flops = self._try_get_flops_fvcore(T, C, H, W)
        thop_macs, thop_flops = self._try_get_flops_thop(T, C, H, W)

        # safety: ensure no THOP leftovers on the live model
        _thop_artifact_cleanup(self.model)

        # Distributions
        times_ms: List[float] = []
        vram_alloc: List[Optional[float]] = []
        vram_reserved: List[Optional[float]] = []

        if use_cuda:
            torch.cuda.empty_cache()

        # slide like real-time
        for i in range(N):
            item = self._build_datas_for_frame(frames_bgr[i])
            buffer.append(item)
            cur_list = list(buffer)
            if len(cur_list) < self.clip_size:
                cur_list = cur_list + [cur_list[-1]] * (self.clip_size - len(cur_list))
            datas, _shape = self._collate_and_stack_clip(cur_list)

            if use_cuda:
                torch.cuda.reset_peak_memory_stats(self.device)

            t_ms = self._time_one_forward(datas, use_cuda=use_cuda)
            times_ms.append(t_ms)

            if use_cuda:
                vram_alloc.append(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
                vram_reserved.append(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2))
            else:
                vram_alloc.append(None)
                vram_reserved.append(None)

        def summarize(vals: List[Optional[float]]) -> Dict[str, Optional[float]]:
            nums = [v for v in vals if v is not None]
            if not nums:
                return {"mean": None, "std": None, "p50": None, "p95": None, "min": None, "max": None}
            arr = np.array(nums, dtype=np.float64)
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        return {
            "model": f"mcgaze",
            "device": str(self.device),
            "dtype": "float32",
            "trials": int(N),
            "input_shape": [T, C, H, W],  # post-pipeline clip tensor (your model gets [img]=[T,C,H,W])
            "parameters_total": total_params,
            "parameters_trainable": trainable_params,
            "flops_per_forward": int(fvcore_flops) if fvcore_flops is not None else None,  # clip-level
            "thop_macs": int(thop_macs) if thop_macs is not None else None,
            "thop_flops": int(thop_flops) if thop_flops is not None else None,
            "latency_ms": {
                "all": times_ms,
                "summary": summarize(times_ms),
            },
            "vram_peak_allocated_mb": {
                "all": vram_alloc,
                "summary": summarize(vram_alloc),
            },
            "vram_peak_reserved_mb": {
                "all": vram_reserved,
                "summary": summarize(vram_reserved),
            },
            "clip_size": self.clip_size,
        }


# ============================= CLI / Save (npz + summary json) =============================

def _save_all_npz(results: Dict[str, Any], out_path: str) -> None:
    """Mimic Gaze3D saver: flatten dicts into arrays and store in a single .npz."""
    save_dict = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    for kkk, vvv in vv.items():
                        save_dict[f"{k}_{kkk}"] = np.array(vvv) if vvv is not None else np.array([])
                elif isinstance(vv, list):
                    save_dict[f"{k}_{kk}"] = np.array(vv)
                else:
                    save_dict[f"{k}_{kk}"] = np.array([vv]) if vv is not None else np.array([])
        elif isinstance(v, list):
            save_dict[k] = np.array(v)
        elif v is None:
            save_dict[k] = np.array([])
        else:
            save_dict[k] = np.array([v])
    np.savez(out_path, **save_dict)
    print(f"Saved results to {out_path}")


def _save_summary_json(results: Dict[str, Any], out_path: str) -> None:
    summary = {
        "model": results["model"],
        "device": results["device"],
        "dtype": results["dtype"],
        "trials": results["trials"],
        "input_shape": results["input_shape"],
        "parameters_total": results["parameters_total"],
        "parameters_trainable": results["parameters_trainable"],
        "flops_per_forward": results["flops_per_forward"],
        "thop_macs": results["thop_macs"],
        "thop_flops": results["thop_flops"],
        "latency_ms": results["latency_ms"]["summary"],
        "vram_peak_allocated_mb": results["vram_peak_allocated_mb"]["summary"],
        "vram_peak_reserved_mb": results["vram_peak_reserved_mb"]["summary"],
        "clip_size": results["clip_size"],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    # ====== EDIT THESE PATHS / SETTINGS ======
    CONFIG_PATH   = "../configs/multiclue_gaze/multiclue_gaze_r50_l2cs.py"
    CHECKPOINT    = "../ckpts/multiclue_gaze_r50_l2cs.pth"
    DEVICE_STR    = "cuda:0" if torch.cuda.is_available() else "cpu"
    CLIP_SIZE     = 7         # match your deployment
    N_FRAMES      = 1000       # how many frames to benchmark
    # raw crops before pipeline (pipeline resizes to 448x448)
    H0, W0        = 448, 448
    # Output names to match your preferred convention
    NPZ_OUT       = f"resource_test_results_mcgaze_clip_size_{CLIP_SIZE}.npz"
    SUMMARY_OUT   = f"resource_test_results_mcgaze_clip_size_{CLIP_SIZE}_summary.json"
    # =========================================

    tester = MCGazeResourceTester(CONFIG_PATH, CHECKPOINT, DEVICE_STR, CLIP_SIZE)

    frames_bgr = (np.random.rand(N_FRAMES, H0, W0, 3) * 255).astype(np.uint8)

    results = tester.run(frames_bgr, trials=N_FRAMES)
    print(json.dumps(results, indent=2))

    # Save full (.npz) + compact summary (.json), mirroring Gaze3D
    _save_all_npz(results, NPZ_OUT)
    _save_summary_json(results, SUMMARY_OUT)
