import rclpy
from rclpy.node import Node
import numpy as np
import time
import os

from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

cwd = "/home/kovan/FaceAndGaze/MCGaze_demo"

def quaternion_rotation_matrix(Q):
    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

class subscriber(Node):

    def __init__(self):
        super().__init__('subscriber')
        self.subscription_helmet = self.create_subscription(
            String,
            'helmet_pos_',
            self.helmet_callback,
            10)
        self.subscription_helmet  # prevent unused variable warning
        
        self.subscription_box = self.create_subscription(
            String,
            'box_pos_',
            self.box_callback,
            10)
        self.subscription_box  # prevent unused variable warning
        
        self.publisher_ = self.create_publisher(MarkerArray, 'gaze_array', 10)
        self.timer = self.create_timer(0.1, self.publish_array)

        # Initialize attributes
        self.gaze_data = np.empty((0, 4))  # 2D array with rows [timestamp, gaze_x, gaze_y, gaze_z]
        self.helmet_pos = np.array([0, 0, 0, 0, 0, 0, 0])
        self.eye_pos = np.array([0, 0, 0, 0, 0, 0, 0])
        self.box_pos = np.array([0, 0, 0])
        self.gaze_vector = np.array([0, 0, 0])

    def helmet_callback(self, msg):
        # Assuming msg.data is a string with comma-separated values
        self.helmet_pos = msg.data.split(',')
        # Convert each item in helmet_pos to an integer
        self.helmet_pos = np.array([float(item) for item in self.helmet_pos])
        rot_mat = quaternion_rotation_matrix(self.helmet_pos[3:7])
        self.eye_pos = self.helmet_pos.copy()
        epos = self.helmet_pos[0:3] + (rot_mat @ np.array([0.07, 0, -0.13]))
        self.eye_pos[0] = epos[0]
        self.eye_pos[1] = epos[1]
        self.eye_pos[2] = epos[2]
        self.gaze_vector = self.box_pos - self.eye_pos[0:3]
        
        temp = np.copy(self.gaze_vector[0])
        self.gaze_vector[0] = self.gaze_vector[1]
        self.gaze_vector[1] = temp

        # Get the current timestamp
        timestamp = self.get_clock().now().to_msg().sec * 1000 + self.get_clock().now().to_msg().nanosec // 1_000_000
        # Append the timestamp and gaze vector as a new row
        new_row = np.array([[timestamp, *self.gaze_vector]])
        self.gaze_data = np.vstack((self.gaze_data, new_row))

    def box_callback(self, msg):
        self.box_pos = msg.data.split(',')
        self.box_pos = np.array([float(item) for item in self.box_pos])
        
    def publish_array(self):
        marker_array = MarkerArray()
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "eye"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.pose.position.x = self.eye_pos[0]
        marker.pose.position.y = self.eye_pos[1]
        marker.pose.position.z = self.eye_pos[2]
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        marker_array.markers.append(marker)
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "helmet"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.pose.position.x = self.helmet_pos[0]
        marker.pose.position.y = self.helmet_pos[1]
        marker.pose.position.z = self.helmet_pos[2]
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        marker_array.markers.append(marker)
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "box"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.pose.position.x = self.box_pos[0]
        marker.pose.position.y = self.box_pos[1]
        marker.pose.position.z = self.box_pos[2]
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        
        marker_array.markers.append(marker)
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "gaze"
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.2
        marker.scale.z = 0.3
        
        marker.points = [
            Point(x=self.eye_pos[0], y=self.eye_pos[1], z=self.eye_pos[2]),
            Point(x=self.eye_pos[0]+self.gaze_vector[0], y=self.eye_pos[1]+self.gaze_vector[1], z=self.eye_pos[2]+self.gaze_vector[2])
        ]
        
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        
        marker_array.markers.append(marker)
        
        self.publisher_.publish(marker_array)

    def save_gaze_data(self):
        # Save gaze data as a .npy file
        np.save(cwd+"/gaze_labels/"+str(int(time.time() * 1000))+".npy", self.gaze_data)


def main(args=None):
    rclpy.init(args=args)
    subscriber_node = subscriber()

    try:
        rclpy.spin(subscriber_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure gaze data is saved before shutdown
        subscriber_node.save_gaze_data()
        if rclpy.ok():  # Check if the context is still valid
            rclpy.shutdown()


if __name__ == '__main__':
    print("Subscribing to helmet_pos_ and box")
    main()
