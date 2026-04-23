import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- 1. ResNet-50 Backbone ---
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # Load ResNet50 with ImageNet weights
        base = models.resnet50(pretrained=True)
        
        # We use everything up to the final pooling layer
        # Output shape: [Batch, 2048, 7, 7]
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# --- 2. Gaze Estimator (MLP) ---
class GazeEstimator(nn.Module):
    def __init__(self):
        super(GazeEstimator, self).__init__()
        # Paper: "output numbers... are 1000 and 2"
        # ResNet-50 outputs 2048 channels
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        # Global Average Pooling: (B, 2048, 7, 7) -> (B, 2048)
        x = torch.mean(x, dim=[2, 3]) 
        x = F.relu(self.fc1(x))
        gaze = self.fc2(x)
        return gaze

# --- 3. Decoder (SA-Module) ---
# Paper: "five-layers SA-Module (N=5)"
class SABlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SABlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        # Upsample 2x
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        shortcut = self.shortcut(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
        
        out = self.conv1(self.relu(self.bn1(out)))
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out

class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        # Paper: "numbers of feature maps are 256, 128, 64, 32, 16"
        
        # Projection from ResNet-50 (2048) to Decoder start (256)
        self.project = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.bn_proj = nn.BatchNorm2d(256)
        
        # 5 Blocks: 7x7 -> 224x224
        self.block1 = SABlock(256, 256) # 7 -> 14
        self.block2 = SABlock(256, 128) # 14 -> 28
        self.block3 = SABlock(128, 64)  # 28 -> 56
        self.block4 = SABlock(64, 32)   # 56 -> 112
        self.block5 = SABlock(32, 16)   # 112 -> 224
        
        self.final = nn.Conv2d(16, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.bn_proj(self.project(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.final(x)
        return x

# --- 4. Main Model Wrapper ---
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature = Backbone()
        self.gazeEs = GazeEstimator()
        self.deconv = Reconstructor()

    def forward(self, x):
        feature = self.feature(x['face'])
        gaze = self.gazeEs(feature)
        recon = self.deconv(feature)
        return gaze, recon

# --- 5. Loss Operations ---
class Gelossop(nn.Module):
    def __init__(self, attentionmap, w1=1, w2=1):
        super(Gelossop, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.l1 = nn.L1Loss()
        self.attentionmap = attentionmap
        self.w1 = w1 # Alpha (Adversarial)
        self.w2 = w2 # Beta (Gaze)
        self.k = 0.75 # Threshold

    def forward(self, gaze, img_recon, label, face):
        # Gaze Loss (L1)
        gaze_loss = self.l1(gaze, label)
        
        # Adversarial Loss
        # Paper: L_backbone = alpha * E[ M * Indicator * (1 - (I - I_hat)^2) ] + ...
        
        # 1. Calculate Pixel Difference (I - I_hat)^2
        pixel_diff = self.mse(img_recon, face) 
        pixel_diff = torch.mean(pixel_diff, dim=1) # Average channels
        
        # 2. Truncated Adversarial Loss Logic (TA-Loss)
        # Calculate L2 norm of difference per image
        diff_norm = torch.norm((face - img_recon).view(face.size(0), -1), p=2, dim=1)
        
        # Indicator function: 1 if diff_norm <= k (Paper phrasing is tricky, but logic is:
        # if reconstruction is BAD (diff > k), we stop penalizing because it's already "purified enough")
        # Paper: "L_adv will be zero if the pixel difference is larger than k"
        mask_k = (diff_norm <= self.k).float().view(-1, 1, 1)

        # 3. Apply Attention Map and Truncation
        # We want to MAXIMIZE reconstruction error.
        # Equivalent to Minimizing (1 - Error).
        # We assume input images are 0-1, so MSE is 0-1.
        
        weighted_loss = self.attentionmap * mask_k * (1 - pixel_diff)
        adv_loss = torch.mean(weighted_loss)

        total_loss = self.w1 * adv_loss + self.w2 * gaze_loss
        return total_loss

class Delossop(nn.Module):
    def __init__(self):
        super(Delossop, self).__init__()
        self.mse = nn.MSELoss() #

    def forward(self, img_recon, face):
        return self.mse(img_recon, face)

