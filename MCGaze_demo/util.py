import numpy as np

# def quaternion_rotation_matrix(Q):
#     # Extract the values from Q
#     q0 = Q[3]
#     q1 = Q[0]
#     q2 = Q[1]
#     q3 = Q[2]
     
#     # First row of the rotation matrix
#     r00 = 2 * (q0 * q0 + q1 * q1) - 1
#     r01 = 2 * (q1 * q2 - q0 * q3)
#     r02 = 2 * (q1 * q3 + q0 * q2)
     
#     # Second row of the rotation matrix
#     r10 = 2 * (q1 * q2 + q0 * q3)
#     r11 = 2 * (q0 * q0 + q2 * q2) - 1
#     r12 = 2 * (q2 * q3 - q0 * q1)
     
#     # Third row of the rotation matrix
#     r20 = 2 * (q1 * q3 - q0 * q2)
#     r21 = 2 * (q2 * q3 + q0 * q1)
#     r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
#     # 3x3 rotation matrix
#     rot_matrix = np.array([[r00, r01, r02],
#                            [r10, r11, r12],
#                            [r20, r21, r22]])
                            
#     return rot_matrix

def order_points_counter_clockwise(points):
    """
    Orders the points in counter-clockwise order.

    Parameters:
        points (array-like): List of points representing a polygon (N, 2 or 3).

    Returns:
        List of points ordered counter-clockwise.
    """
    # Convert to a NumPy array for convenience
    points = np.array(points)

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the angle of each point with respect to the centroid
    # `np.arctan2` gives angles in radians from -π to π
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points based on their angles
    sorted_indices = np.argsort(angles)

    # Return the points in the sorted order
    return points[sorted_indices]
