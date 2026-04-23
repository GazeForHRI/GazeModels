import os
import numpy as np
import torch
from torchvision import transforms
from gaze_estimation import GazeModel
import model_resnet50
from typing import List, Tuple

class PureGazeOnGaze360(GazeModel):
    def __init__(self, weights_path: str, device="cuda:0", rectification=False):
        super().__init__(rectification=rectification)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.model = self._load_model()
        
        # Preprocessing to match training/eval logic:
        # 1. ToPILImage: Handles numpy array input
        # 2. Resize: Ensure 224x224 (Base class usually handles this, but good for safety)
        # 3. ToTensor: Converts [0, 255] uint8 -> [0.0, 1.0] float and [H, W, C] -> [C, H, W]
        # This matches the reader_gaze360.py logic used during your training.
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # standard ImageNet Normalization, do not use for PureGaze, does not work since it updates ResNet during training.
        ])

    def get_model_name(self) -> str:
        if self.rectification:
            return "puregaze_gaze360_rectification"
        return "puregaze_gaze360"

    def _load_model(self):
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"Model weights not found at {self.weights_path}")

        # Use the ResNet-50 version
        net = model_resnet50.Model()
        
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        
        # Handle state_dict wrapping if present (robustness for different save methods)
        if 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
            
        net.to(self.device)
        net.eval()
        return net

    @staticmethod
    def pitch_yaw_to_unit_vector(pitch, yaw):
        # Convert radians to 3D vector
        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)
        return np.array([x, y, z], dtype=np.float64)

    def estimate_from_crops(self, head_crops: List[np.ndarray], timestamps: np.ndarray):
        gaze_with_ts = []
        valid_indices = []

        batch_size = 128
        N = len(head_crops)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = head_crops[start:end]
            ts_batch = timestamps[start:end]

            # Preprocess converts to [0-1] float tensors (RGB)
            input_tensor = torch.stack([self.preprocess(img) for img in batch]).to(self.device)

            with torch.no_grad():
                # model_resnet50 returns (gaze, reconstruction)
                outputs, _ = self.model({"face": input_tensor})
                outputs = outputs.cpu().numpy()

            for i, (ts, pred) in enumerate(zip(ts_batch, outputs)):
                if pred.shape[0] != 2:
                    continue
                
                # Based on eval_puregaze.py: 
                # p_yaw, p_pitch = pred[:, 0], pred[:, 1]
                yaw = pred[0]
                pitch = pred[1]
                
                gaze_vec = self.pitch_yaw_to_unit_vector(pitch, yaw)
                
                # Keep consistent basis transformation with the pipeline
                gaze_vec = self.transform_gaze_to_custom_basis(gaze_vec)
                
                gaze_with_ts.append([float(ts)] + list(gaze_vec))
                valid_indices.append(start + i)

        return np.array(gaze_with_ts, dtype=np.float64), np.array(valid_indices, dtype=np.int32)

    @staticmethod
    def transform_gaze_to_custom_basis(vec: np.ndarray) -> np.ndarray:
        vec = vec / np.linalg.norm(vec)
        return np.array([vec[2], -vec[0], -vec[1]])  # [z, x, y]

