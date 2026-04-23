from abc import ABC, abstractmethod
import numpy as np
import cv2
from tqdm import tqdm
from enum import Enum
import os

class GazeModel(ABC):
    def __init__(self, rectification=False, isRGB = False, headcrop_padding = 0):
        self.rectification = rectification
        self.isRGB = isRGB
        if(not rectification):
            self.headcrop_padding = headcrop_padding

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the name of the gaze estimation model (used when saving estimation files).
        """
        pass
    @abstractmethod
    def estimate_from_crops(self, head_crops: list[np.ndarray], timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate gaze given a list of cropped head regions and timestamps.
        Returns:
            - np.ndarray: [N, 4] array with [timestamp, gaze1, gaze2, gaze3]
            - np.ndarray: [M] valid frame indices (subset of 0...N)
        """
        pass
    
    @staticmethod
    def transform_gaze_to_custom_basis(vec: np.ndarray) -> np.ndarray:
        return vec

    def estimate(self, rgb_video: list[np.ndarray], timestamps: np.ndarray, bboxes: np.ndarray, output_size=(224, 224), padding=False, exp_dir = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract head crops using bboxes, then delegate to `estimate_from_crops`.

        Args:
            rgb_video (List[np.ndarray]): full frames.
            timestamps (np.ndarray): timestamps in seconds or ms.
            bboxes (np.ndarray): array of shape (N, 6) with [x1, y1, x2, y2, id, frame_idx]
            output_size (tuple): resize target (default: 224x224)
            padding (bool): whether to pad to square
            exp_dir: experiment directory: Only give a valid path if you want preprocessing

        Returns:
            Tuple[np.ndarray, np.ndarray]: [timestamp, gaze1, gaze2, gaze3], valid indices
        """
        # ----------------------------------------------- START OF RECTIFICATION --------------------------------------------
        # WILL TAKE RGB_VIDEO FRAMES AND APPLY RECTIFICATION ACCORDINGLY
        if(self.rectification):
            try:
                if(exp_dir == None):
                    raise GazeEstimationValidationException("Experiment directory must be specified in order to apply rectification!!")
                import data_rectification as dpc
                from data_matcher import match_regular_to_regular
                import config
                eye_positions_path = os.path.join(exp_dir, "eye_positions.npy")
                head_poses_path = os.path.join(exp_dir, "head_poses.npy")
                target_positions_path = os.path.join(exp_dir, "target_positions.npy")
                camera_intrinsics_path = os.path.join(exp_dir, "camera_intrinsics.npy")

                camera_poses_path = os.path.join(exp_dir, "camera_poses.npy")
                

                if not os.path.exists(eye_positions_path) or not os.path.exists(head_poses_path) or not os.path.exists(camera_intrinsics_path) or not os.path.exists(target_positions_path):
                    print("os.path.exists(eye_positions_path):", os.path.exists(eye_positions_path))
                    print("os.path.exists(head_poses_path):", os.path.exists(head_poses_path))
                    print("os.path.exists(camera_intrinsics_path):", os.path.exists(camera_intrinsics_path))
                    print("os.path.exists(target_positions):", os.path.exists(target_positions_path))
                    raise GazeEstimationValidationException("Missing input files")
                
                if not os.path.exists(camera_poses_path):
                    raise GazeEstimationValidationException(f"Missing input file: {camera_poses_path}")
                
                eye_positions = np.load(eye_positions_path, allow_pickle=True) # [Timestamp, x, y, z]
                head_poses = np.load(head_poses_path, allow_pickle=True) # [Timestamp, Transformation_Matrix_Flattened]
                target_positions = np.load(target_positions_path, allow_pickle=True) # [Timestamp, x, y, z]
                camera_intrinsics = np.load(camera_intrinsics_path, allow_pickle=True).item() # {'fx', 'fy', 'cx', 'cy', 'distortion_coeffs'}
                camera_poses = np.load(camera_poses_path, allow_pickle=True)  # [Timestamp, 16]
                
                _, index_array = match_regular_to_regular(
                        tuple1=(timestamps.reshape(-1,1), 30.30),
                        tuple2=(eye_positions, 5318008),
                        max_match_diff_ms=2000,
                        stability_tolerance_ms=20.0,
                        stability_window_size=5
                        )
                eye_positions = eye_positions[index_array]
                head_poses = head_poses[index_array]
                target_positions = target_positions[index_array]

                camera_poses = camera_poses[index_array]

                # Prepare outputs
                eye_positions_cam = eye_positions.copy()
                head_poses_cam = head_poses.copy()
                target_positions_cam = target_positions.copy()
                headrotvectors = np.zeros((head_poses.shape[0], 3), dtype=np.float64)

                R_world_to_cam = np.zeros((head_poses.shape[0], 3, 3), dtype=np.float64) #storage for frame trasformations per frame

                # Transform each sample to the camera frame
                for i in range(head_poses.shape[0]):
                    # camera pose: world_T_cam (4x4) is the inverse of cam_T_world
                    cam_T_world = camera_poses[i, 1:].reshape(4, 4)
                    world_T_cam = np.linalg.inv(cam_T_world)

                    R_world_to_cam[i] = world_T_cam[:3, :3]

                    # --- EYE POSITION: world -> camera
                    pw = np.append(eye_positions[i, 1:4], 1.0)              # [x,y,z,1] in world
                    pc = world_T_cam @ pw                                   # camera-frame homogeneous
                    eye_positions_cam[i, 1:4] = pc[:3]

                    # --- TARGET POSITION: world -> camera
                    pw = np.append(target_positions[i, 1:4], 1.0)              # [x,y,z,1] in world
                    pc = world_T_cam @ pw                                   # camera-frame homogeneous
                    target_positions_cam[i, 1:4] = pc[:3]

                    # --- HEAD POSE: world -> camera (pose matrix)
                    head_T_world = head_poses[i, 1:].reshape(4, 4)          # head pose in world
                    head_T_cam = world_T_cam @ head_T_world                 # head pose in camera
                    head_poses_cam[i, 1:] = head_T_cam.flatten()

                    # Extract rotation vector (Rodrigues) from head pose in camera frame
                    R = head_T_cam[:3, :3]
                    rvec, _ = cv2.Rodrigues(R)
                    headrotvectors[i] = rvec.reshape(3)

                # Camera intrinsics for dpc.norm
                def _as_camparams(ci):
                    if not isinstance(ci, dict):
                        raise TypeError(f"Expected dict for camera intrinsics, got {type(ci)}")
                    fx = float(ci["fx"]); fy = float(ci["fy"]); cx = float(ci["cx"]); cy = float(ci["cy"])
                    dist = np.array(ci.get("distortion_coeffs", np.zeros(5)), dtype=np.float64).ravel()
                    K = np.array([[fx, 0.0, cx],
                                [0.0, fy, cy],
                                [0.0, 0.0, 1.0]], dtype=np.float64)
                    return {"mtx": K, "dist": dist}

                camera = _as_camparams(camera_intrinsics)

                # Frame Preprocess Loop
                # Will take every frame 
                # --- Normalize every frame with per-frame geometry --------------------------

                normalized_frames = []
                M_mats = []  # store rectification matrices per frame
                for i, frame in enumerate(tqdm(rgb_video, desc="Gaze estimation with model: " + self.get_model_name() + "(RECTIFICATION)")):
                    if frame is None or frame.size == 0:
                        # keep a black placeholder to maintain indexing
                        normalized_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                        print("ERROR: ASYNC DETECTED AT PREPROCESS STEP")
                        continue

                    def convert_to_opencv_basis(v):
                        v = np.asarray(v)
                        assert v.shape == (3,), f"Expected shape (3,), got {v.shape}"
                        # Apply basis transformation
                        return np.array([-v[1], -v[2], v[0]])
                    def flip_yaw_180(rvec):
                        """
                        Rotates the rotation vector rvec by 180° around the Y-axis.
                        This flips the yaw angle (turns the head around to the opposite direction).
                        """
                        # Convert rvec to rotation matrix
                        R, _ = cv2.Rodrigues(rvec)

                        # 180° rotation matrix around Y-axis
                        R_flip_yaw = np.array([
                            [-1, 0,  0],
                            [ 0, 1,  0],
                            [ 0, 0, -1]
                        ])
                        # Apply the yaw flip
                        R_new = R_flip_yaw @ R
                        # Convert back to rotation vector
                        rvec_new, _ = cv2.Rodrigues(R_new)
                        return rvec_new.flatten()
                    def rotate_rvec_z_axis(rvec, degrees=-90):
                        """
                        Rotates the rotation vector (rvec) around the Z-axis by a specified angle (in degrees).
                        Useful to correct roll or reorient head pose.
                        
                        Parameters:
                            rvec: np.array, shape (3,), Rodrigues rotation vector
                            degrees: float, angle to rotate around Z-axis (negative = clockwise)

                        Returns:
                            new_rvec: np.array, shape (3,), rotated Rodrigues vector
                        """
                        # Convert angle to radians
                        angle_rad = np.deg2rad(degrees)

                        # Rotation matrix around Z-axis
                        R_z = np.array([
                            [np.cos(angle_rad), -np.sin(angle_rad), 0],
                            [np.sin(angle_rad),  np.cos(angle_rad), 0],
                            [0,                 0,                 1]
                        ])

                        # Convert rvec to rotation matrix
                        R_orig, _ = cv2.Rodrigues(rvec)

                        # Apply rotation: first R_z, then original
                        R_new = R_z @ R_orig

                        # Convert back to rvec
                        new_rvec, _ = cv2.Rodrigues(R_new)
                        return new_rvec.flatten()
                    
                    center_i = convert_to_opencv_basis(eye_positions_cam[i, 1:4])*1000     # (3,) in CAMERA frame
                    target_i = convert_to_opencv_basis(target_positions_cam[i, 1:4])*1000  # (3,) in CAMERA frame
                    rvec_i   = rotate_rvec_z_axis(flip_yaw_180(convert_to_opencv_basis(headrotvectors[i])),-90)           # (3,) Rodrigues in CAMERA frame

                    # dpc.norm expects: center (3,), target (3,), headrotvec (3,), imsize (2,), camparams (3x3)
                    N = dpc.norm(center=center_i,
                                gazetarget=target_i,
                                headrotvec=rvec_i,
                                imsize=(224, 224),
                                camparams=camera.get("mtx"))
                                        
                    #save the 3×3 M matrix used to normalize this frame
                    M_mats.append(N.M_mat.copy())

                    # Warp full frame into normalized view (224x224)

                    norm_img = N.GetImage(frame)   # cv2.warpPerspective (CPU)

                    if self.isRGB:
                        norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)

                    normalized_frames.append(norm_img)

                    cv2.imshow("Normalized Frames", norm_img) # For debugging
                    cv2.waitKey(-1)

                gaze_norm, valid_indices = self.estimate_from_crops(normalized_frames, timestamps)
                
                # gaze_norm: [N, 4] = [timestamp, gx, gy, gz] in *normalized camera* frame
                # valid_indices: which frames are valid

                t0 = self.transform_gaze_to_custom_basis(np.array([1., 0., 0.]))
                t1 = self.transform_gaze_to_custom_basis(np.array([0., 1., 0.]))
                t2 = self.transform_gaze_to_custom_basis(np.array([0., 0., 1.]))

                C = np.column_stack([t0/np.linalg.norm(t0),
                                    t1/np.linalg.norm(t1),
                                    t2/np.linalg.norm(t2)])   # 3x3
                C_inv = np.linalg.inv(C)

                gaze_camera = gaze_norm.copy()
                gw = []
                for j, idx in enumerate(valid_indices):
                    # 1) normalized-frame gaze, untransfrom custom basis
                    g_custom = gaze_norm[j, 1:4] # [z, -x, -y] Some sort of basis transform
                    # We need to inverse this transform without knowing the "transform_gaze_to_custom_basis"
                    g_norm = C_inv @ g_custom
                    g_norm /= np.linalg.norm(g_norm) + 1e-12

                    # 2) back to original camera frame: g_cam ∝ M^{-1} * g_norm
                    M = M_mats[idx]
                    try:
                        M_inv = np.linalg.inv(M)
                    except np.linalg.LinAlgError:
                        # Fallback in pathological cases
                        M_inv = np.linalg.pinv(M)

                    g_cam = M_inv @ g_norm
                    g_cam = g_cam / (np.linalg.norm(g_cam) + 1e-8)

                    # 4) store back into result
                    gw.append(g_cam)
                    g_cam = self.transform_gaze_to_custom_basis(g_cam)
                    gaze_camera[j, 1:4] = g_cam
                
                
                gw = gaze_camera[:, 1:4]; gn = gaze_norm[:, 1:4]
                cos_theta = np.clip(
                    np.sum(gw * gn, axis=1) /
                    (np.linalg.norm(gw, axis=1) * np.linalg.norm(gn, axis=1) + 1e-8),
                    -1.0, 1.0
                )
                angles_deg = np.degrees(np.arccos(cos_theta))
                print("MSE:", np.mean((gw-gn)**2), "Mean Angular Diff (deg):", (np.mean(angles_deg)))
                
                return gaze_camera, valid_indices

            except Exception as e:
                print(f"Couldn't Apply Rectification: {e}")

        # ----------------------------------------------- END OF RECTIFICATION ----------------------------------------------

        # ------------------------------------------------ START OF CROPPING ------------------------------------------------
        else:
            frame_to_crop = {int(b[5]): b for b in bboxes}
            head_crops = []

            for i, frame in enumerate(tqdm(rgb_video, desc="Gaze estimation with model: " + self.get_model_name() + "(HEAD CROPPING)")):
                if i not in frame_to_crop:
                    raise Exception(f"Missing bbox for frame {i}")
                x1, y1, x2, y2 = map(int, frame_to_crop[i][:4])

                padd = self.headcrop_padding # Works ass a padding
                crop = frame[max(y1-padd,0):min(y2+padd,frame.shape[0]), max(x1-padd,0):min(x2+padd,frame.shape[1])]

                if self.isRGB:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # Convert to RGB

                if crop.size == 0:
                    crop = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
                elif padding:
                    crop = self.resize_with_padding(crop, output_size)
                else:
                    crop = cv2.resize(crop, output_size)

                cv2.imshow("Crop", crop)  # Debugging line to visualize the crop
                cv2.waitKey(-1)  # Allow OpenCV to process the window events
                head_crops.append(crop)

            
            return self.estimate_from_crops(head_crops, timestamps)

        # ------------------------------------------------ END OF CROPPING ---------------------------------------------------

    def save_estimation(self, gaze_directions: np.ndarray, valid_indices: np.ndarray, output_dir: str, append_model_name_to_output_path=True):
        base_output_path = output_dir
        if append_model_name_to_output_path:
            base_output_path = os.path.join(base_output_path, self.get_model_name())

        # Ensure directory exists
        os.makedirs(base_output_path, exist_ok=True)

        # Construct full paths
        gaze_directions_path = os.path.join(base_output_path, "gaze_directions.npy")
        valid_indices_path = os.path.join(base_output_path, "gaze_directions_indices.npy")

        # Save data
        np.save(gaze_directions_path, gaze_directions)
        np.save(valid_indices_path, valid_indices)

    def resize_with_padding(self, image, target_size=(224, 224)):
        h, w = image.shape[:2]
        target_w, target_h = target_size

        if h == 0 or w == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        scale = min(target_w / w, target_h / h)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))

        pad_w = target_w - resized.shape[1]
        pad_h = target_h - resized.shape[0]

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded

class GazeEstimationValidationException(Exception):
    pass

class GazeEstimationStatus(Enum):
    SUCCESSFUL = 1
    UNEXPECTED_ERROR = 2
    VALIDATION_FAILED = 3