import os
import numpy as np
import torch
from torchvision import transforms
from gaze_estimation import GazeModel


class PureGaze(GazeModel):
    def __init__(self, weights_path: str, device="cuda:0", rectification=False):
        super().__init__(rectification=rectification)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.model = self._load_model()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_model_name(self) -> str:
        if self.rectification:
            return "puregaze_rectification"
        return "puregaze"

    def _load_model(self):
        from model.model import Model

        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"Model weights not found at {self.weights_path}")

        net = Model()
        state_dict = torch.load(self.weights_path, map_location=self.device)
        net.load_state_dict(state_dict)
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
                outputs, _ = self.model({"face": input_tensor}, require_img=False)
                outputs = outputs.cpu().numpy()

            #print(f"Processed batch {start}-{(end) // batch_size}, pitch:{outputs.mean(0)[0]:.2f} - yaw: {outputs.mean(0)[1]:.2f}")

            for i, (ts, pred) in enumerate(zip(ts_batch, outputs)):
                if pred.shape[0] != 2:
                    continue
                # ETH-XGaze gives (pitch, yaw), but PureGaze expects (yaw, pitch)
                eth_pitch, eth_yaw = pred
                yaw, pitch = eth_yaw, eth_pitch
                gaze_vec = self.pitch_yaw_to_unit_vector(pitch, yaw)
                gaze_vec = self.transform_gaze_to_custom_basis(gaze_vec)
                gaze_with_ts.append([float(ts)] + list(gaze_vec))
                valid_indices.append(start + i)

        return np.array(gaze_with_ts, dtype=np.float64), np.array(valid_indices, dtype=np.int32)

    @staticmethod
    def transform_gaze_to_custom_basis(vec: np.ndarray) -> np.ndarray:
        vec = vec / np.linalg.norm(vec)
        return np.array([vec[2], -vec[0], -vec[1]])  # [z, x, y]
