import numpy as np
import torch
import torch.nn as nn

from l2cs.utils import prep_input_numpy, getArch
from gaze_estimation import GazeModel

class L2CSNet(GazeModel):
    def __init__(self, weights_path: str, arch="ResNet50", device="cpu",rectification=False, vanilla_preprocess = False):
        super().__init__(rectification=rectification)
        self.device = torch.device(device)
        self.model = getArch(arch, 90)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(self.device)

    def get_model_name(self) -> str:
        if self.rectification:
            return "l2cs_rectification"
        return "l2cs"

    def estimate_from_crops(self, head_crops: list[np.ndarray], timestamps: np.ndarray):
        gaze_with_ts = []
        valid_indices = []

        batch_size = 64  # You can tune this (try 32 or even 16 if OOM persists)
        N = len(head_crops)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = np.stack(head_crops[start:end])
            ts_batch = timestamps[start:end]

            img_batch = prep_input_numpy(batch, self.device)

            with torch.no_grad():
                gaze_pitch, gaze_yaw = self.model(img_batch)

                pitch_pred = self.softmax(gaze_pitch)
                yaw_pred = self.softmax(gaze_yaw)

                pitch = torch.sum(pitch_pred * self.idx_tensor, dim=1) * 4 - 180
                yaw = torch.sum(yaw_pred * self.idx_tensor, dim=1) * 4 - 180

                pitch = pitch.cpu().numpy() * np.pi / 180.0
                yaw = yaw.cpu().numpy() * np.pi / 180.0

            for i, (ts, p, y) in enumerate(zip(ts_batch, pitch, yaw)):
                gaze_dir = self.pitch_yaw_to_unit_vector(p, y)
                gaze_dir = self.transform_l2cs_to_custom_basis(gaze_dir)
                gaze_with_ts.append([float(ts)] + list(gaze_dir))
                valid_indices.append(start + i)

        return np.array(gaze_with_ts, dtype=np.float64), np.array(valid_indices, dtype=np.int32)

    @staticmethod
    def pitch_yaw_to_unit_vector(pitch, yaw):
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        return np.array([x, y, z], dtype=np.float64)

    @staticmethod
    def transform_l2cs_to_custom_basis(vec: np.ndarray) -> np.ndarray:
        vec = vec / np.linalg.norm(vec)
        return np.array([vec[2], -vec[1], -vec[0]])  # [z, x, y]
