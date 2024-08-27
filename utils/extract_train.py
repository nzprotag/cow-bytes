import deeplabcut
import os
import pandas as pd
import glob
import argparse
import numpy as np
from features import columns_

# Function Definitions
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
# Function to calculate the distance ratio

def calculate_ratio(poses_x, poses_y, idx1, idx2, idx3, idx4):
    dist1 = euclidean_distance(poses_x.iloc[:, idx1], poses_y.iloc[:, idx1], poses_x.iloc[:, idx2], poses_y.iloc[:, idx2])
    dist2 = euclidean_distance(poses_x.iloc[:, idx3], poses_y.iloc[:, idx3], poses_x.iloc[:, idx4], poses_y.iloc[:, idx4])

    # Calculate the ratio and handle division by zero
    try:
        ratio = dist1 / dist2
    except ZeroDivisionError:
        ratio = 0

    return ratio

def process_folder(full_folder_path, subset, class_type, salient, folder, path_config_file):
    def rename_path(x):
        return os.path.join(subset, class_type, salient, folder, str(x))

    full_folder_path = os.path.join(os.getcwd(), full_folder_path)
    deeplabcut.analyze_time_lapse_frames(path_config_file, full_folder_path, frametype='.jpg', save_as_csv=True)
    
    csv_file = glob.glob(os.path.join(full_folder_path, '*.csv'))[0]
    df = pd.read_csv(csv_file).drop([0, 1]).reset_index(drop=True)

    # Extract the x and y coordinates
    poses_x = df.iloc[:, 1::3].astype(float)
    poses_y = df.iloc[:, 2::3].astype(float)
    
    ratio_df = pd.DataFrame()
    for col in columns_:
        pose_1, pose_2 = [list(map(int, x.split('_')[1:])) for x in col.split('_to_')]
        ratio_df[col] = calculate_ratio(poses_x, poses_y, int(pose_1[0]), int(pose_1[1]), int(pose_2[0]), int(pose_2[1]))

    # Rename columns
    ratio_df.insert(0, 'scorer', df['scorer'].apply(rename_path))
    ratio_df.insert(1, 'class', [class_type] * len(df))
    
    return ratio_df

def extract_train(args):
    extracted_dir = args.extracted_dir
    path_config_file = args.path_config_file
    subsets = args.subsets.split(',')
    classes = args.classes.split(',')
    salients = args.salients.split(',')
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    final_df = None
    for subset in subsets:
        for class_type in classes:
            for salient in salients:
                salient_path = os.path.join(extracted_dir, subset, class_type, salient)
                folders = [f for f in os.listdir(salient_path) if os.path.isdir(os.path.join(salient_path, f))]
                for folder in folders:
                    full_folder_path = os.path.join(salient_path, folder)
                    df = process_folder(full_folder_path, subset, class_type, salient, folder, path_config_file)
                    final_df = pd.concat([final_df, df], ignore_index=True) if final_df is not None else df

                    non_jpg_files = [f for f in glob.glob(os.path.join(full_folder_path, '*')) if not f.lower().endswith('.jpg')]
                    for file in non_jpg_files:
                        try:
                            os.remove(file)
                            print(f"Deleted: {file}")
                        except Exception as e:
                            print(f"Error deleting {file}: {e}")

        final_df.to_csv(os.path.join(os.getcwd(), save_dir, f"{subset}.csv"), index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process pose data and calculate ratios.')
    parser.add_argument('--extracted_dir', type=str, required=True, help='Path to the extracted data directory.')
    parser.add_argument('--path_config_file', type=str, required=True, help='Path to the DeepLabCut config file.')
    parser.add_argument('--subsets', type=str, required=True, help='Comma-separated list of subsets.')
    parser.add_argument('--classes', type=str, required=True, help='Comma-separated list of classes.')
    parser.add_argument('--salients', type=str, required=True, help='Comma-separated list of salients.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save the processed CSV files.')

    args = parser.parse_args()
    extract_train(args)