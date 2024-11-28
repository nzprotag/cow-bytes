#!/bin/bash

cd ..

# Download the dataset
mkdir -p data
cd data
kaggle datasets download -d sttaseen/bitecount
unzip bitecount.zip -d ./BiteCount

# Extract salient frames
cd ../utils
python split_salient.py ./../data/BiteCount/ bitecount_poserac.csv

# # Extract the landmarks
# python extract_train.py --extracted_dir ./../data/BiteCount/extracted \
#                         --path_config_file ./../dlc/CowBytes-Single-Sadat-2024-06-30/config.yaml \
#                         --subsets train \
#                         --classes cow_bite \
#                         --salients salient1,salient2 \
#                         --save_dir ./../data/BiteCount/annotation_pose/

# # Extract the landmarks
# python extract_test.py  --path_config_file ./../dlc/CowBytes-Single-Sadat-2024-06-30/config.yaml \
#                         --test_destination ./../data/BiteCount/test_poses  \
#                         --test_csv ./../data/BiteCount/annotation/cow_bite_links_1407_clipped.csv \
#                         --video_source ./../data/BiteCount/video 