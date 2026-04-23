from __future__ import annotations
import os
import numpy as np
import torch
from torchvision import transforms
from gaze_estimation import GazeModel
import cv2
import gtools

class GazeTRGaze360(GazeModel):
    def __init__(self, weights_path="checkpoints/GazeTR-H-Gaze360.pt", device="cuda:0", rectification=False):
        super().__init__(rectification=rectification)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.model = self._load_model()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
        ])

    def get_model_name(self) -> str:
        if self.rectification:
            return "gazetr_gaze360_rectification"
        return "gazetr_gaze360"

    def _load_model(self):
        # GazeTR's model.py should be under 'GazeTR/model.py'
        from model import Model

        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.weights_path}")

        net = Model()
        state_dict = torch.load(self.weights_path, map_location=self.device)
        net.load_state_dict(state_dict)  # pretrained model key
        net.to(self.device)
        net.eval()
        return net

    @staticmethod
    def pitch_yaw_to_unit_vector(pitch, yaw):
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        return np.array([x, y, z], dtype=np.float64)

    def estimate_from_crops(self, head_crops: list[np.ndarray], timestamps: np.ndarray):
        gaze_with_ts = []
        valid_indices = []

        batch_size = 32
        N = len(head_crops)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = head_crops[start:end]
            ts_batch = timestamps[start:end]

            input_tensor = torch.stack([self.preprocess(img) for img in batch]).to(self.device)

            with torch.no_grad():
                outputs = self.model({"face": input_tensor})
                outputs = outputs.cpu().numpy()

            for i, (ts, pred) in enumerate(zip(ts_batch, outputs)):
                if pred.shape[0] != 2:
                    continue
                
                yaw = pred[0]   # pred[0] is Yaw in Gaze360
                pitch = pred[1] # pred[1] is Pitch in Gaze360
                
                gaze_vec = self.pitch_yaw_to_unit_vector(pitch, yaw)
                
                gaze_vec = self.transform_gaze_to_custom_basis(gaze_vec)

                gaze_with_ts.append([float(ts)] + list(gaze_vec))
                valid_indices.append(start + i)

        return np.array(gaze_with_ts, dtype=np.float64), np.array(valid_indices, dtype=np.int32)
    
    @staticmethod
    def visualize_gaze_on_img(img, gaze_vector):
        img = img.copy()
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        scale = 40
        endpoint = (
            int(center[0] + scale * gaze_vector[0]),
            int(center[1] + scale * gaze_vector[1])
        )
        cv2.arrowedLine(img, center, endpoint, (0, 255, 0), 2)
        return img


    @staticmethod
    def transform_gaze_to_custom_basis(vec: np.ndarray) -> np.ndarray:
        vec = vec / np.linalg.norm(vec)
        return np.array([vec[2], -vec[0], -vec[1]])  # [z, x, y]
