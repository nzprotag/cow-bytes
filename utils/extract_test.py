import deeplabcut
import os
import pandas as pd
import glob
import argparse
import numpy as np
import shutil
from features import columns_

def extract_test(args):
    test_destination = os.path.join(os.getcwd(), args.test_destination)
    valid_df = pd.read_csv(args.test_csv)
    os.makedirs(test_destination, exist_ok=True)

    for video_name in valid_df['name']:
        source = os.path.join(args.video_source, video_name)
        shutil.copy(source, test_destination)

    deeplabcut.analyze_videos(args.path_config_file, test_destination,save_as_csv=True)

    # Read csv file and process it
    csv_files = glob.glob(os.path.join(test_destination, '*.csv'))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Dropping the first two rows
        df = df.drop([0, 1]).reset_index(drop=True)
        
        # Drop the first column
        df.drop(['scorer'], axis=1, inplace=True)
        
        """
        OPTIONAL: Calculate the ratios.
        """
        # Extract the x and y coordinates
        poses_x = df.iloc[:, ::3].astype(float)
        poses_y = df.iloc[:, 1::3].astype(float)

        # Function to calculate Euclidean distance
        def euclidean_distance(x1, y1, x2, y2):
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Function to calculate the distance ratio
        def calculate_ratio(df, idx1, idx2, idx3, idx4):
            dist1 = euclidean_distance(poses_x.iloc[:, idx1], poses_y.iloc[:, idx1], poses_x.iloc[:, idx2], poses_y.iloc[:, idx2])
            dist2 = euclidean_distance(poses_x.iloc[:, idx3], poses_y.iloc[:, idx3], poses_x.iloc[:, idx4], poses_y.iloc[:, idx4])

            # Calculate the ratio and handle division by zero
            try:
                ratio = dist1 / dist2
            except ZeroDivisionError:
                ratio = 0

            return ratio

        # List to store the ratio columns
        ratio_df = pd.DataFrame()

        # Calculate the ratios
        for col in columns_:
            # Extract pose indices from the column name
            parts = col.split('_to_')
            pose_1 = parts[0].split('_')[1:]
            pose_2 = parts[1].split('_')[1:]

            # Calculate the ratio
            ratio_df[col] = calculate_ratio(df, int(pose_1[0]), int(pose_1[1]), int(pose_2[0]), int(pose_2[1]))
            
        df = ratio_df
        
        """
        END OF OPTIONAL PART
        """

        # Convert to np_array and save it
        np_array = df.to_numpy(dtype=float)

        # Extract the video_name
        video_name = os.path.basename(csv_file).split('DLC')[0]

        np.save(os.path.join(test_destination, video_name), np_array)

    # Delete any temp files
    all_files = glob.glob(os.path.join(test_destination, '*'))

    # Filter out files that are not npy
    non_npy_files = [f for f in all_files if not f.lower().endswith('.npy')]

    # Delete non-JPG files
    for file in non_npy_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process pose data and calculate ratios.')
    parser.add_argument('--path_config_file', type=str, required=True, help='Path to the DeepLabCut config file.')
    parser.add_argument('--test_destination', type=str, required=True, help='Path to save the processed CSV test files.')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to CSV test annotaions.')
    parser.add_argument('--video_source', type=str, required=True, help='Path to CSV test annotaions.')


    args = parser.parse_args()
    extract_test(args)