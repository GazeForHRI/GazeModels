import numpy as np
import os
import glob
import time

# Define the directory where files are saved
cwd = os.getcwd()  # Or the path you used in your script

# use "gaze_estimations" or "gaze_labels" or "gaze_errors" as the directory
def load_np_file(directory):
    save_directory = os.path.join(cwd, directory)

    # Get the list of .npy files
    files = glob.glob(os.path.join(save_directory, "*.npy"))

    if files:
        # Get the latest file based on modification time
        latest_file = max(files, key=os.path.getmtime)
        print(f"Loading the latest file: {latest_file}")

        return np.load(latest_file)
    else:
        raise Exception("No .npy files found in the directory")

def gaze_error(est,label):
    est = np.array(est)
    label = np.array(label)
    
    # Normalize the vectors
    est_norm = est / np.linalg.norm(est)
    label_norm = label / np.linalg.norm(label)
    
    # Calculate angular error (in degrees)
    dot_product = np.dot(est_norm, label_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical issues with acos
    angular_error = np.degrees(np.arccos(dot_product))
    
    # Calculate Euclidean distance error
    euclidean_error = np.linalg.norm(est - label)
    
    return angular_error,euclidean_error

def save_gaze_error(errors,filename):
    errors = np.array(errors)
    
    # Calculate mean and standard deviation for angular and Euclidean errors
    angular_mean = np.mean(errors[:, 0])
    angular_std = np.std(errors[:, 0])
    euclidean_mean = np.mean(errors[:, 1])
    euclidean_std = np.std(errors[:, 1])
    
    stats = f"{angular_mean}\n{angular_std}\n{euclidean_mean}\n{euclidean_std}\n"
    # Save stats to a file
    with open(filename+".txt", "w") as file:
        file.write(stats)
        # Save results to a file
    # Save entire data as numpy array
    np.save(filename+".npy", errors)
        
    print(f"Gaze error results saved to {filename}")
    print(f"Mean: angular error, gaze error: {angular_mean:.4f} deg, {euclidean_mean:.4f}")
    print(f"Standard deviation: angular error, gaze error: {angular_std:.4f} deg, {euclidean_std:.4f}")

def process_and_save_errors(filename):
    # Load the saved estimations and labels
    gaze_estimations = load_np_file("gaze_estimations")  # Each row: [timestamp, gaze_x, gaze_y, gaze_z]
    gaze_labels = load_np_file("gaze_labels")           # Each row: [timestamp, gaze_x, gaze_y, gaze_z]
    
    # Prepare a list to store results
    error_results = []
    
    # Match estimations and labels by timestamp (within 200ms)
    for est in gaze_estimations:
        timestamp = est[0]
        est_vector = est[1:]
        
        # Find labels within the 200ms window
        time_diff = np.abs(gaze_labels[:, 0] - timestamp)
        label_indices = np.where(time_diff <= 200)[0]  # Indices of labels within the 200ms window
        
        if len(label_indices) > 0:
            # If multiple labels fall in the window, choose the closest one
            closest_label_idx = label_indices[np.argmin(time_diff[label_indices])]
            label_vector = gaze_labels[closest_label_idx, 1:]  # Get the gaze vector part of the label

            # Calculate errors
            angular_error, euclidean_error = gaze_error(est_vector, label_vector)

            if (True):
                print("timestamp: ", timestamp)
                print("estimation: ", est_vector)
                print("label: ", label_vector)
                print("error result: ", [timestamp, angular_error, euclidean_error])
            
            # Append to results
            error_results.append([timestamp, angular_error, euclidean_error])
    

    # Convert results to a NumPy array    

    # Ensure it's always a 2D array (matrix)
    if len(error_results) == 0:
        print('no data point to calculate error from:')
        error_results = np.zeros((0, 3))  # Create an empty 2D matrix with shape (0, 0)
    else:
        error_results = np.array(error_results)
 
    # Save the results to a file
    np.save(filename, error_results)
    print(f"Error results saved to {filename}")
    
    # Calculate mean and standard deviation for angular and Euclidean errors
    angular_errors = error_results[:, 1]
    euclidean_errors = error_results[:, 2]
    
    mean_angular = np.mean(angular_errors)
    std_angular = np.std(angular_errors)
    mean_euclidean = np.mean(euclidean_errors)
    std_euclidean = np.std(euclidean_errors)
    
    stats = f"{mean_angular}\n{std_angular}\n{mean_euclidean}\n{std_euclidean}\n"
    # Save stats to a file (stats saved with .txt extension whereas errors saved with .npy extension)
    with open(filename + ".txt", "w") as file:
        file.write(stats)
    
    print(f"Mean Angular Error: {mean_angular:.2f} degrees")
    print(f"Standard Deviation Angular Error: {std_angular:.2f} degrees")
    print(f"Mean Euclidean Error: {mean_euclidean:.4f}")
    print(f"Standard Deviation Euclidean Error: {std_euclidean:.4f}")
    
    return error_results, mean_angular, std_angular, mean_euclidean, std_euclidean

    

if __name__ == "__main__":
    # a = load_np_file("gaze_labels")
    # print(a.shape, a)
    # exit()
    #add current timestamp to the filename, file will be under the gaze_errors directory
    filename = cwd + "/gaze_errors/" + str(int(time.time() * 1000))
    process_and_save_errors(filename)