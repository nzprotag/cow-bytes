#!/bin/bash

cd ..

# Download the dataset
mkdir -p data
cd data
kaggle datasets download -d sttaseen/bitecount-a
unzip bitecount-a.zip -d ./

# Extract salient frames
cd ../utils
python split_salient.py ./../data/BiteCount-A/ cow_bite_links_1407_clipped.csv
