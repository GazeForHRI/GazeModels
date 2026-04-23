# measure_flops_params.py
import torch
from fvcore.nn import FlopCountAnalysis

# --- your codebase ---
from model import modules
from model.model import Model  # adjust if your file name/path differs

torch.set_grad_enabled(False)

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

# Instantiate modules using YOUR implementation
backbone = modules.resnet50(pretrained=True).eval()
gaze_es  = modules.ResGazeEs().eval()
deconv   = modules.ResDeconv(modules.BasicBlock).eval()
full     = Model().eval()

# Dummy input that matches your pipeline (NCHW RGB)
x = torch.randn(1, 3, 224, 224)

# ---- Param counts ----
bb_total, bb_train = count_params(backbone)
gz_total, gz_train = count_params(gaze_es)
dc_total, dc_train = count_params(deconv)
fm_total, fm_train = count_params(full)

print("="*60)
print("[Params]")
print(f"feature (backbone): total={bb_total:,} | trainable={bb_train:,}")
print(f"gazeEs           : total={gz_total:,} | trainable={gz_train:,}")
print(f"deconv           : total={dc_total:,} | trainable={dc_train:,}")
print(f"FULL Model       : total={fm_total:,} | trainable={fm_train:,}")

# ---- FLOPs: feature maps once, reuse for heads ----
with torch.no_grad():
    feats = backbone(x)  # whatever your resnet50(modules) returns; used by gazeEs/deconv

# Sanity check shape (gazeEs & deconv expect a 4D feature map)
if feats.dim() != 4:
    raise RuntimeError(f"Backbone returned shape {tuple(feats.shape)}; expected 4D conv features.")

# Per-module FLOPs
flops_feature = FlopCountAnalysis(backbone, x).total()
flops_gaze    = FlopCountAnalysis(gaze_es, feats).total()
flops_deconv  = FlopCountAnalysis(deconv, feats).total()

print("="*60)
print("[FLOPs | per-module]")
print(f"feature (backbone): {flops_feature/1e9:.3f} GFLOPs")
print(f"gazeEs            : {flops_gaze/1e9:.3f} GFLOPs")
print(f"deconv            : {flops_deconv/1e9:.3f} GFLOPs")

# Full model FLOPs with require_img=True (gaze + deconv)
inp_dict = {"face": x}
flops_full_with_img  = FlopCountAnalysis(full, (inp_dict,)).total()

# Full model FLOPs with require_img=False (gaze only)
class FullNoImg(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x_in):
        # call your Model but force require_img=False
        gaze, _ = self.base(x_in, require_img=False)
        return gaze

full_no_img = FullNoImg(full).eval()
flops_full_no_img = FlopCountAnalysis(full_no_img, (inp_dict,)).total()

print("="*60)
print("[FLOPs | end-to-end]")
print(f"FULL (require_img=True) : {flops_full_with_img/1e9:.3f} GFLOPs")
print(f"FULL (require_img=False): {flops_full_no_img/1e9:.3f} GFLOPs")

# Optional: Print a tiny breakdown (top ops) for the gaze head
try:
    from fvcore.nn import flop_count_table
    print("="*60)
    print("[gazeEs FLOP breakdown]")
    print(flop_count_table(FlopCountAnalysis(gaze_es, feats)))
except Exception:
    pass
