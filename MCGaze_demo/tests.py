from util import order_points_counter_clockwise
import numpy as np

def test_order_points_counter_clockwise():
    print("Testing test_order_points_counter_clockwise...")
    # Test case 1: Rectangle
    points = [
        [1, 1],  # Top-right
        [-1, 1],  # Top-left
        [-1, -1],  # Bottom-left
        [1, -1],  # Bottom-right
    ]
    expected = [
        [-1, -1],  # Bottom-left
        [1, -1],   # Bottom-right
        [1, 1],    # Top-right
        [-1, 1],   # Top-left
    ]
    result = order_points_counter_clockwise(points)
    assert np.allclose(result, expected), f"Test case 1 failed: {result}"

    # Test case 2: Points already in counter-clockwise order
    points = [
        [-1, -1],  # Bottom-left
        [1, -1],   # Bottom-right
        [1, 1],    # Top-right
        [-1, 1],   # Top-left
    ]
    expected = [
        [-1, -1],  # Bottom-left
        [1, -1],   # Bottom-right
        [1, 1],    # Top-right
        [-1, 1],   # Top-left
    ]
    result = order_points_counter_clockwise(points)
    assert np.allclose(result, expected), f"Test case 2 failed: {result}"

    # Test case 3: Triangle
    points = [
        [0, 1],  # Top
        [-1, -1],  # Bottom-left
        [1, -1],  # Bottom-right
    ]
    expected = [
        [-1, -1],  # Bottom-left
        [1, -1],   # Bottom-right
        [0, 1],    # Top
    ]
    result = order_points_counter_clockwise(points)
    assert np.allclose(result, expected), f"Test case 3 failed: {result}"

    # Test case 4: Points in a random order
    points = [
        [1, -1],  # Bottom-right
        [0, 1],  # Top
        [-1, -1],  # Bottom-left
    ]
    expected = [
        [-1, -1],  # Bottom-left
        [1, -1],   # Bottom-right
        [0, 1],    # Top
    ]
    result = order_points_counter_clockwise(points)
    assert np.allclose(result, expected), f"Test case 4 failed: {result}"

    # Test case 5: Square rotated
    points = [
        [0, 1],   # Top-center
        [1, 0],   # Right-center
        [0, -1],  # Bottom-center
        [-1, 0],  # Left-center
    ]
    expected = [
        [0, -1],  # Bottom-center
        [1, 0],   # Right-center
        [0, 1],   # Top-center
        [-1, 0],  # Left-center
    ]
    result = order_points_counter_clockwise(points)
    assert np.allclose(result, expected), f"Test case 5 failed: {result}"
    
    
    # Test case 6: Triangle
    points = [
        [0, 1],
        [3, -4],
        [-1, -1],
    ]
    expected = [
        [3, -4],
        [0, 1],
        [-1, -1],
    ]
    result = order_points_counter_clockwise(points)
    assert np.allclose(result, expected), f"Test case 6 failed: {result}"


    print("All test cases passed!")


if __name__ == "__main__":
    test_order_points_counter_clockwise()
