import pandas as pd
import os

# --- Configuration ---
INPUT_FILE = '/home/kovan/USTA/gaze_test/July-Experiments/flattened_blink_annotations/flattened_blink_annotations_by_deniz_04_half1/path_mapping.csv'
OUTPUT_FILE = '/home/kovan/USTA/gaze_test/July-Experiments/flattened_blink_annotations/flattened_blink_annotations_by_deniz_04_half1/path_mapping.csv'
COLUMN_TO_FIX = 'exp_dir_relpath'

def fix_paths(input_filename, output_filename, column_name):
    """
    Reads a CSV, replaces backslashes with forward slashes in a specified 
    column, and saves the result to a new CSV file.
    """
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return

    print(f"Reading data from '{input_filename}'...")
    
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(input_filename)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the CSV.")
        print(f"Available columns are: {list(df.columns)}")
        return

    print(f"Replacing '\\\\' with '/' in column '{column_name}'...")
    
    # Apply the string replacement to the specified column
    # The first argument is '\\\\' because the backslash itself must be escaped
    df[column_name] = df[column_name].astype(str).str.replace('\\', '/', regex=False)

    # Save the modified DataFrame to the new CSV file
    df.to_csv(output_filename, index=False)

    print(f"Successfully fixed paths and saved the new file to '{output_filename}'.")

if __name__ == "__main__":
    fix_paths(INPUT_FILE, OUTPUT_FILE, COLUMN_TO_FIX)