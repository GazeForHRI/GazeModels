import cv2
import numpy as np
import torch
from demo_real_time import infer_gaze, visualize_gaze_mcgaze
from yolo_head.detect_and_return_frame_results import detect
import time
import pyrealsense2 as rs
from skspatial.objects import Line, Plane
import numpy as np

from socket_server import send_gaze_vector
from util import order_points_counter_clockwise


# Define the socket file
SOCKET_FILE = '/tmp/usta_sockets_gaze.sock'

# @DEPRECATED all wrt world frame
EYE = [-0.21,0.32,1.13]# tail of the gaze vector
# @DEPRECATEDquaternions to represent the orientation of the camera link 
CAMERA_LINK_QUAT = [-0.1408, 0.019, 0.989, 0.034]
# @DEPRECATEDTable plane:
TABLE_NORMAL = [0,0,1]
TABLE_POINT = [-0.155,-0.485,0.72]
# set to true to publish gaze vector to the socket (which will then be published to ROS). if the necessary ROS node is not working and you just want to run this file on its own,
# you need to set this to false to avoid a socket error.
PUBLISH_GAZE = False # you can set this to true if you want to publish the gaze vector to the socket. if you do not need to publish, and need to resolve the socket error (in send_gaze_vector client.connect(socket_path)), set this to false.
RGB_RESOLUTION = (1920,1080)

cwd = "/home/kovan/FaceAndGaze/MCGaze_demo"

head_crops = []
head_crops_path = "demo_video_head_crops.npy"
gaze_vectors = []
gaze_vectors_path = "demo_video_gaze_vectors.npy"

# Initialize an empty array for gaze estimations with timestamps
# Each row will be [timestamp_ms, gaze_x, gaze_y, gaze_z]
gaze_estimations = np.empty((0, 4), dtype=np.float64)

def compute_gaze_target(gd, gt, table_points):
    """
    Compute the gaze target location (tar) as the intersection of the table plane and the gaze line.

    Parameters:
        gd (array-like): Gaze direction vector (3D, shape (3,))
        gt (array-like): Gaze line tail (3D, shape (3,))
        table_points (array-like): 4 points of the table plane (shape (4,3))

    Returns:
        tuple:
            found (bool): True if a valid gaze target is found, False otherwise.
            tar (np.ndarray or None): The gaze target location (3D) if found, otherwise None.
    """
    # Test cases (no longer valid since q has been replaced by table_points):

    # Case 1, no intersection (empty set):
    # Should return False, None
    # gd = [1, 1, 1]  # Gaze direction
    # gt = [0, 0, 1]  # Gaze tail
    # n = [1, -1, 0]  # Table normal
    # q = [10, 0, 0]  # A point on the table plane

    # Case 2, the intersection is the whole line:
    # Should return False, None
    # gd = [1, 1, 1]  # Gaze direction
    # gt = [0, 0, 0]  # Gaze tail
    # n = [1, -1, 0]  # Table normal
    # q = [0, 0, 0]  # A point on the table plane

    # Case 3, intersection is a point, but an invalid one:
    # Should return False, None
    # gd = [1, 1, 1]  # Gaze direction
    # gt = [2, 2, 2]  # Gaze tail
    # n = [1, 1, 1]  # Table normal
    # q = [0, 0, 0]  # A point on the table plane

    # Case 4, intersection is a valid point:
    # Should return True, [0,0,0]
    # gd = [1, 1, 1]  # Gaze direction
    # gt = [-2, -2, -2]  # Gaze tail
    # n = [1, 1, 1]  # Table normal
    # q = [0.0, 0.0, 0.0]  # A point on the table plane

    try:
        table_points = order_points_counter_clockwise(table_points)
        table_center = np.mean(table_points, axis=0)
        table_n = np.cross(table_points[0] - table_center, table_points[1] - table_center)
        
        # Create Line and Plane objects using the provided parameters
        line = Line(point=gt, direction=gd)
        plane = Plane(point=table_points[0], normal=table_n)

        # Attempt to find the intersection point
        intersection_point = plane.intersect_line(line)

        # Convert the Point object to a NumPy array
        tar = np.array(intersection_point)

        # if gaze target is on the opposite direction of the gaze vector, return False.
        if np.dot(tar - gt, gd) <= 0.0:
            return False, None
        
        # Check if the intersection point is inside the table polygon                
        first_cross = None
        
        for i in range(len(table_points) - 1):
            cross = np.cross(table_points[i+1] - table_points[i], table_center - table_points[i+1])
            
            if first_cross is None:
                first_cross = cross
            else:
                # if the cross product is ever in the opposite direction of the first cross product, return False since that implies the point is outside the table.
                #Note: the cross product can be either in the exacy same direction or in the exact opposite direction, never in between.
                if np.dot(first_cross, cross) < 0:
                    return False, None

        return True, tar

    except ValueError:
        # No valid intersection found
        return False, None


def quaternion_rotation_matrix(Q):
    """
    Converts a quaternion into a 3x3 rotation matrix.
    
    Parameters:
        Q: A list or numpy array representing the quaternion [q1, q2, q3, q0],
           where q1, q2, q3 are the vector part, and q0 is the scalar part.
    
    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    # Extract the values from Q
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
    q0 = Q[3]
    
    # First row of the rotation matrix
    r00 = 1 - 2 * (q2**2 + q3**2)
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 1 - 2 * (q1**2 + q3**2)
    r12 = 2 * (q2 * q3 - q0 * q1)
    
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 1 - 2 * (q1**2 + q2**2)
    
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    
    return rot_matrix


# Simulate accessing the gaze vector and adding a timestamp
def add_gaze_estimation(gaze_vector):
    global gaze_estimations

    # Get the current timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000)

    # Ensure gaze_vector is a flat array
    if isinstance(gaze_vector, np.ndarray):
        gaze_vector = gaze_vector.squeeze()  # Removes dimensions of size 1
    elif isinstance(gaze_vector, list):
        gaze_vector = np.array(gaze_vector).squeeze()

    # Validate the size of gaze_vector
    if gaze_vector.shape != (3,):
        raise ValueError(f"gaze_vector must have exactly 3 elements, but got shape {gaze_vector.shape}")

    # Combine the timestamp and gaze vector into a single row
    new_row = np.array([[timestamp_ms, *gaze_vector]], dtype=np.float64)

    # Append the new row to the gaze_estimations array
    gaze_estimations = np.vstack((gaze_estimations, new_row))


DOES_CALCULATE_GAZE_ERROR = False
#from torchvision.transforms.functional import crop

def visualize_crop(cropped_image):
    cropped = cropped_image.cpu().detach().numpy()
    cv2.imwrite("cropped_test.png", cropped[0][0])

def find_head_area(frame):
    # Make sure frame dimensions are multiples of 32 (required for YOLO)
    h, w = frame.shape[2], frame.shape[3]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32

    if (new_h, new_w) != (h, w):
        frame = frame.cpu().detach().numpy()  # Convert tensor to NumPy array
        frame = np.squeeze(frame, axis=0)  # Remove batch dimension if needed
        frame = np.transpose(frame, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        frame = cv2.resize(frame, (new_w, new_h))
        
        # Convert back: from NumPy array (H, W, C) -> PyTorch Tensor (C, H, W)
        frame = torch.tensor(frame, dtype=torch.float32)  # Convert to tensor
        frame = torch.permute(frame, (2, 0, 1))  # Convert (H, W, C) -> (C, H, W)
        frame = frame.unsqueeze(0)  # Add batch dimension (1, C, H, W)
        frame = frame.to(torch.device("cuda:0"))  # Move to GPU if needed

    head_detection_result = detect(frame)
    
    if len(head_detection_result) > 0:
        head_detection_result = head_detection_result[0].int()
    else:
        head_detection_result = None

    return head_detection_result



def save_latency(data, file_name):
    # Convert observations to a NumPy array
    observations_array = np.array(data)

    # Calculate statistics
    mean_time = np.mean(observations_array)
    std_dev_time = np.std(observations_array)

    stats = f"{mean_time}\n{std_dev_time}\n"
    # Save stats to a file
    with open(file_name+".txt", "w") as file:
        file.write(stats)
    # Save entire data as numpy array
    np.save(file_name+".npy", observations_array)

    # Print summary
    print(f"Latency results saved to {file_name}")
    print(f"Mean latency: {mean_time:.4f} seconds")
    print(f"Standard deviation of latency: {std_dev_time:.4f} seconds")

def process_cropped_frame(cropped_frame, frame_buffer, clip_size=-1):
    """
    Directly process a cropped head frame (already cropped by the client).

    Args:
        cropped_frame (np.ndarray): Head crop (BGR image).
        frame_buffer (list): Temporal buffer.
        clip_size (int): Size of the clip for temporal models.

    Returns:
        tuple: (clip, head_detection_result, cropped_frame)
    """
    if cropped_frame is None or cropped_frame.size == 0:
        return None, None, None

    try:
        # Assume this is already a head crop (cropped RGB frame from the client)
        h, w, _ = cropped_frame.shape
        l = max(h, w) // 2  # Use half of the size as approximate scale

        clip = infer_gaze(cropped_frame, l, frame_buffer, clip_size=clip_size)

        # Fake head_detection_result just for visualization compatibility
        dummy_box = [0, 0, h, w]  # [y1, x1, y2, x2]
        return clip, dummy_box, cropped_frame

    except Exception as e:
        print("process_frame error:", e)
        return None, None, None


def process_frame(frame, frame_buffer, clip_size=-1):
    frame_tensor = torch.tensor(frame).to(torch.device("cuda:0")).float()
    frame_tensor = torch.permute(frame_tensor, (2, 0, 1)).unsqueeze(0)
    
    head_detection_result = find_head_area(frame_tensor)
    cropped_numpy = None  # Default value
    
    if head_detection_result is not None:
        head_center = [
            int(head_detection_result[1] + head_detection_result[3]) // 2,
            int(head_detection_result[0] + head_detection_result[2]) // 2
        ]
        l = int(max(
            head_detection_result[3] - head_detection_result[1],
            head_detection_result[2] - head_detection_result[0]
        ) * 0.8)
        
        w, h, _ = frame.shape
        cropped_numpy = frame[
            max(0, head_center[0] - l): min(head_center[0] + l, w),
            max(0, head_center[1] - l): min(head_center[1] + l, h),
            :
        ]
        
        clip = infer_gaze(cropped_numpy, l, frame_buffer, clip_size=clip_size)
    else:
        clip = None
    
    return clip, head_detection_result, cropped_numpy


def format_array(arr):
    np.set_printoptions(precision=2, suppress=True)

    for row in arr:
        formatted_row = ["{:.2f}".format(x) for x in row]
        print("[" + "  ".join(formatted_row) + "]")

def process_camera(case_name='none'):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, RGB_RESOLUTION[0], RGB_RESOLUTION[1], rs.format.bgr8, 30)

    pipeline.start(config)
    frame_buffer = []

    while True:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = np.asanyarray(frames.get_color_frame().get_data())
            
            # Process frame for gaze estimation
            clip, head_detection_result, cropped_frame = process_frame(color_frame, frame_buffer)
            

            # If gaze estimation succeeded, visualize the result on the original frame
            if clip is not None:
                gaze_vector = clip["gaze_p"][-1]
                gaze_vector_unit = gaze_vector[0,0,:] / np.linalg.norm(gaze_vector[0,0,:])
                if PUBLISH_GAZE:
                    # the message param comprises the gaze vector and timestamp in ms
                    send_gaze_vector(SOCKET_FILE, f"{gaze_vector_unit[2]} {gaze_vector_unit[0]} {gaze_vector_unit[1]} {int(time.time() * 1000)}")# we swap and x and y values to match the world frame in ROS
                print("Gaze in camera frame: ", gaze_vector_unit)# unit vector
                visualized_frame = visualize_gaze_mcgaze(color_frame.copy(), gaze_vector, head_detection_result)
                head_crops.append(np.array(color_frame))
                gaze_vectors.append(np.array([gaze_vector_unit[2], gaze_vector_unit[0], gaze_vector_unit[1]]))
            else:
                visualized_frame = color_frame.copy()

            # Ensure the cropped frame is valid
            if cropped_frame is not None and cropped_frame.size > 0:
                # Show the combined image (Gaze estimation + Raw frame)
                cv2.imshow("Cropped Image", cropped_frame)

            # Show the combined image (Gaze estimation + Raw frame)
            cv2.imshow("Gaze Visualization", visualized_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        except KeyboardInterrupt:
            break

    np.save(gaze_vectors_path, gaze_vectors)
    np.save(head_crops_path, head_crops)
    pipeline.stop()
    cv2.destroyAllWindows()

            
def process_video(video_path):
    """
    Process the WebM video file for real-time gaze estimation.

    Args:
        video_path: Path to the WebM video file.
    """
    # Open the WebM video file
    frame_buffer = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            #frame = cv2.imread("/home/kovan/FaceAndGaze/MCGaze_demo/single_frame/0.jpg")
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not ret:
                print("End of video.")
                break

            # Preprocess the frame for inference
            clip, head_detection_result = process_frame(frame, frame_buffer)
            if clip["gaze_p"]:
                gaze_vector = clip["gaze_p"][-1]  # Get the latest gaze vector

                # Visualize gaze on the original frame
                visualized_frame = visualize_gaze_mcgaze(frame.copy(), gaze_vector, head_detection_result)

                # Show the visualized frame
                cv2.imshow("Gaze Estimation", visualized_frame)
                #cv2.imwrite("result.png", visualized_frame)
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()



def visualize_npy_video(npy_path, wait=33, window_name="RGB Video"):
    """
    Play a NumPy video file (saved as (N, H, W, 3) array or object array of frames).

    Args:
        npy_path (str): Path to the .npy file containing RGB frames
        wait (int): Delay between frames in milliseconds (approx. 33 ms for 30 FPS).
        window_name (str): Name of the OpenCV window.
    """
    frames = np.load(npy_path, allow_pickle=True)
    print("video shape:", frames.shape)

    # Case 1: Standard video (N, H, W, 3)
    if isinstance(frames, np.ndarray) and frames.ndim == 4 and frames.shape[-1] == 3:
        print(f"Loaded full video with {len(frames)} frames.")
    # Case 2: Ragged face list (dtype=object)
    elif isinstance(frames, np.ndarray) and frames.dtype == object:
        print(f"Loaded {len(frames)} frames.")
    else:
        print("Invalid frame format:", frames.shape, frames.dtype)
        return

    for i, frame in enumerate(frames):
        if frame is None or not isinstance(frame, np.ndarray):
            continue
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(wait)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    


# Example usage
if __name__ == "__main__":
    # gaze_estimations = np.load(gaze_vectors_path)
    # print("gaze_estimations shape:", gaze_estimations.shape)
    # format_array(gaze_estimations)
    # visualize_npy_video(head_crops_path, wait=33, window_name="head_crops")
    # exit()
    # Path to the WebM video
    # video_path = "video_1.mp4"
    # Process the WebM video
    #process_video(video_path)
    process_camera()