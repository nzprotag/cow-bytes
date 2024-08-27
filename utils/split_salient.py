import os
import cv2
import pandas as pd
import numpy as np
import argparse

"""
Modified from PoseRAC. This script will separate the salient frames from the videos.
"""

def _annotation_transform(root_dir, annotation_name):
    train_type = 'train'
    video_dir = os.path.join(root_dir, 'video')
    label_filename = os.path.join(root_dir, 'annotation', annotation_name)
    train_save_dir = os.path.join(root_dir, 'extracted')
    save_dir = os.path.join(train_save_dir, train_type)
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    df = pd.read_csv(label_filename)
    file2label = {}
    num_idx = 3
    
    for i in range(0, len(df)):
        filename = df.loc[i, 'name']
        action_type = df.loc[i, 'type']
        label_tmp = df.values[i][num_idx:].astype(np.float64)
        label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
        s1_tmp = label_tmp[::2]
        s2_tmp = label_tmp[1::2]
        file2label[filename] = [s1_tmp, s2_tmp, action_type]

    for video_name in file2label:
        video_path = os.path.join(video_dir, video_name)
        print('Processing video:', video_path)
        cap = cv2.VideoCapture(video_path)
        frames = []

        if cap.isOpened():
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frames.append(frame)
        cap.release()

        s1_label, s2_label, action_type = file2label[video_name]
        count = 0
        
        # Saving salient1 frames
        for frame_index in s1_label:
            if frame_index >= len(frames):
                continue
            frame_ = frames[frame_index]
            sub_s1_save_dir = os.path.join(save_dir, action_type, 'salient1', video_name)
            if not os.path.isdir(sub_s1_save_dir):
                os.makedirs(sub_s1_save_dir)
            save_path = os.path.join(sub_s1_save_dir, f'{count}.jpg')
            cv2.imwrite(save_path, frame_)
            count += 1

        # Saving salient2 frames
        for frame_index in s2_label:
            if frame_index >= len(frames):
                continue
            frame_ = frames[frame_index]
            sub_s2_save_dir = os.path.join(save_dir, action_type, 'salient2', video_name)
            if not os.path.isdir(sub_s2_save_dir):
                os.makedirs(sub_s2_save_dir)
            save_path = os.path.join(sub_s2_save_dir, f'{count}.jpg')
            cv2.imwrite(save_path, frame_)
            count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract key frames from videos based on pose-level annotations.')
    parser.add_argument('root_dir', type=str, help='Root directory containing the videos and annotations.')
    parser.add_argument('annotation_name', type=str, help='Name of the annotation CSV file.')
    
    args = parser.parse_args()
    _annotation_transform(args.root_dir, args.annotation_name)
