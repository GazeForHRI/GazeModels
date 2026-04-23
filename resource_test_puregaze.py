#!/usr/bin/env python3
import json
import time
import statistics as stats
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn


# ------------------------------- FLOPs helpers (no adapter) -------------------------------

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


def compute_flops_without_adapter_fvcore(model: nn.Module,
                                         device: torch.device,
                                         sample_1chw: torch.Tensor) -> Optional[int]:
    """
    Return FLOPs (int) for a single forward using fvcore, or None if unavailable.
    Monkey-patches model.forward so it accepts a Tensor and returns only the Tensor output.
    """
    from fvcore.nn import FlopCountAnalysis

    sample_1chw = sample_1chw.to(device, non_blocking=True).to(torch.float32)
    model.eval()

    orig_forward = model.forward
    try:
        def proxy_forward(x, *args, **kwargs):
            out, _ = orig_forward({"face": x}, require_img=False)
            return out

        model.forward = proxy_forward  # monkey-patch

        with torch.no_grad():
            fca = FlopCountAnalysis(model, (sample_1chw,))
            flops = int(fca.total())
        return flops if flops > 0 else None
    finally:
        model.forward = orig_forward  # always restore


def compute_macs_without_adapter_thop(model: nn.Module,
                                      device: torch.device,
                                      sample_1chw: torch.Tensor) -> Optional[int]:
    """
    Return MACs (int) for a single forward using THOP, or None if unavailable.
    Monkey-patches model.forward so it accepts a Tensor and returns only the Tensor output.
    """
    from thop import profile

    sample_1chw = sample_1chw.to(device, non_blocking=True).to(torch.float32)
    model = model.to(device).eval()

    orig_forward = model.forward
    try:
        def proxy_forward(x, *args, **kwargs):
            out, _ = orig_forward({"face": x}, require_img=False)
            return out

        model.forward = proxy_forward

        with torch.no_grad():
            macs, _ = profile(model, inputs=(sample_1chw,), verbose=False)
        return int(macs) if macs and macs > 0 else None
    finally:
        model.forward = orig_forward  # always restore


# ----------------------------------- Wrapper -----------------------------------

class PureGazeWrapper:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.eval().to(device)
        self.device = device

    def _count_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return int(total), int(trainable)

    def _try_get_flops(self, sample_1chw: torch.Tensor) -> Optional[int]:
        """
        Try fvcore (FLOPs). If that fails, try THOP (MACs) and convert to FLOPs≈2×MACs.
        Returns FLOPs (int) or None.
        """
        # fvcore path
        if _has_fvcore():
            try:
                flops = compute_flops_without_adapter_fvcore(self.model, self.device, sample_1chw)
                if flops is not None and flops > 0:
                    return int(flops)
            except Exception as e:
                print(f"[FLOPs] fvcore failed: {e}")

        # thop fallback (MACs -> FLOPs≈2×MACs)
        if _has_thop():
            try:
                macs = compute_macs_without_adapter_thop(self.model, self.device, sample_1chw)
                if macs is not None and macs > 0:
                    return int(macs * 2)
            except Exception as e:
                print(f"[FLOPs] thop failed: {e}")

        return None

    def _time_one(self, x: torch.Tensor, use_cuda: bool) -> float:
        """
        Return latency in milliseconds for a single forward pass (batch_size=1).
        Uses CUDA events if on GPU for better precision.
        """
        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.device)
            starter.record()
            _ = self.model({"face": x}, require_img=False)
            ender.record()
            torch.cuda.synchronize(self.device)
            return float(starter.elapsed_time(ender))  # ms
        else:
            t0 = time.perf_counter()
            _ = self.model({"face": x}, require_img=False)
            t1 = time.perf_counter()
            return (t1 - t0) * 1000.0

    def resource_test(self,
                      trials: int = 100,
                      shape: Tuple[int, int, int] = (3, 224, 224),
                      dtype: torch.dtype = torch.float32,
                      seed: int = 1234) -> Dict:
        """
        Run resource measurement for batch_size=1 over `trials` independent runs.
        Returns dict with distributions for inference time and VRAM.
        """
        C, H, W = shape
        torch.manual_seed(seed)
        use_cuda = (self.device.type == "cuda" and torch.cuda.is_available())
        self.model.eval()
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True  # allow best algos for fixed input size

        # Static metrics
        total_params, trainable_params = self._count_params()
        sample = torch.randn(1, C, H, W, device=self.device, dtype=dtype)
        flops = self._try_get_flops(sample.detach().clone().cpu())

        # Warm-up
        warm = torch.zeros(1, C, H, W, dtype=dtype, device=self.device)
        for _ in range(5):
            _ = self.model({"face": warm}, require_img=False)
        if use_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        # Distributions
        times_ms = []
        vram_alloc = []
        vram_reserved = []

        for _ in range(trials):
            x = torch.randn(1, C, H, W, device=self.device, dtype=dtype)

            if use_cuda:
                torch.cuda.reset_peak_memory_stats(self.device)

            t_ms = self._time_one(x, use_cuda=use_cuda)
            times_ms.append(t_ms)

            if use_cuda:
                vram_alloc.append(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
                vram_reserved.append(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2))
            else:
                vram_alloc.append(None)
                vram_reserved.append(None)

        def summarize(vals):
            vals_num = [v for v in vals if v is not None]
            if not vals_num:
                return {"mean": None, "std": None, "p50": None, "p95": None, "min": None, "max": None}
            return {
                "mean": float(stats.fmean(vals_num)),
                "std": float(stats.pstdev(vals_num)) if len(vals_num) > 1 else 0.0,
                "p50": float(torch.tensor(vals_num).median().item()),
                "p95": float(torch.quantile(torch.tensor(vals_num), 0.95).item()),
                "min": float(min(vals_num)),
                "max": float(max(vals_num)),
            }

        results = {
            "model": "puregaze",
            "device": str(self.device),
            "dtype": str(dtype).replace("torch.", ""),
            "trials": int(trials),
            "input_shape": [1, C, H, W],
            "parameters_total": total_params,
            "parameters_trainable": trainable_params,
            "flops_per_forward": int(flops) if flops is not None else None,  # FLOPs for 1×224×224
            "latency_ms": {
                "all": times_ms,  # raw per-trial latencies
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
        }
        return results


# ----------------------------------- CLI / Demo -----------------------------------

if __name__ == "__main__":
    import numpy as np

    # ==== Basic config variables (edit here) ====
    weights_path = "/home/kovan/FaceAndGazePureGaze/checkpoint/Res50_PureGaze_ETH.pt"
    device_str   = "cuda:0" if torch.cuda.is_available() else "cpu"
    trials       = 1000
    height       = 224
    width        = 224
    dtype_str    = "float32"   # options: "float32", "float16", "bfloat16"
    seed         = 1234
    out_path     = "resource_test_results_puregaze.npz"
    summary_out_path = "resource_test_results_puregaze_summary.json"
    # ============================================

    from gaze_model_puregaze import PureGaze

    device = torch.device(device_str)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[dtype_str]

    model_api = PureGaze(
        weights_path=weights_path,
        device=str(device),
        rectification=False
    )
    wrapper = PureGazeWrapper(model=model_api.model, device=device)

    # Run tests
    results = wrapper.resource_test(
        trials=trials,
        shape=(3, height, width),
        dtype=dtype,
        seed=seed
    )

    # Optional THOP MACs quick check
    thop_macs = None
    thop_flops = None
    if _has_thop():
        macs = compute_macs_without_adapter_thop(wrapper.model, device,
                                                 torch.randn(1, 3, height, width))
        if macs is not None:
            thop_macs = int(macs)
            thop_flops = int(2 * macs)
            print(f"THOP MACs: {thop_macs} -> FLOPs≈ {thop_flops}")

    # Add THOP results into results dict
    results["thop_macs"] = thop_macs
    results["thop_flops"] = thop_flops

    # Print nicely
    import json
    print(json.dumps(results, indent=2))

    # Save everything as .npz
    # Flatten lists to numpy arrays for saving
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
    # summary = {
    #     "latency_ms": results["latency_ms"]["summary"],
    #     "vram_peak_allocated_mb": results["vram_peak_allocated_mb"]["summary"],
    #     "vram_peak_reserved_mb": results["vram_peak_reserved_mb"]["summary"],
    #     "thop_macs": results["thop_macs"],
    #     "thop_flops": results["thop_flops"],
    # }

    # with open(summary_out_path, "w") as f:
    #     json.dump(summary, f, indent=2)
    # print(f"Saved summary to {summary_out_path}")
    summary = results
    summary["latency_ms"] = results["latency_ms"]["summary"]
    summary["vram_peak_allocated_mb"] = results["vram_peak_allocated_mb"]["summary"]
    summary["vram_peak_reserved_mb"] = results["vram_peak_reserved_mb"]["summary"]

    with open(summary_out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_out_path}")

