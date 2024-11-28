# cow-bytes

This repo contains the setup and results for our experiments on cow bite counting as a part of our final year project, sponsored by [ProTag](https://www.protag.co.nz/). 

Clone recursively:
```
git clone --recurse https://github.com/prototaip-134/cow-bytes.git
```

<div align="center">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/user-attachments/assets/6faeca0d-c8ad-45de-88cf-e0a53546d930"
  width=700
  height=auto
  ><br>
    <p style="font-size:1.5vw;">Collage of BiteCount</p>
  </div>
</div>

The BiteCount dataset used can be found on [Kaggle](https://www.kaggle.com/datasets/sttaseen/bitecount). A demo of our model can be found under the [experiments](experiments/10_demo).


## Directory
**After** downloading and extracting the dataset, the directory should look like below:
```
scrambmix
├── data
│   ├── BiteCount
│   │   └── ...
│   └── bitecount.zip
├── dlc
├── experiments
│   ├── 1_decision_tree
│   └── 2_random_forest
│   └── ...
├── README.md
├── setup
│   └── setup.sh
│   └── ...
└── utils
    └── ...
```

In-depth information about what each experiment does can be found in their respective notebooks.

## Requirements
### Setting up a conda environment

#### Install MiniConda
The following instructions are for Linux. For other operating systems, download and install from [here](https://docs.conda.io/en/latest/miniconda.html).
```
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
 "Miniconda3.sh"
```
Install the .sh file.
```
bash Miniconda3.sh
```
Remove the installer:
```
rm Miniconda3.sh
```
#### Creating a virtual environment
Run the following commands to create a virtual environment and to activate it:
```
conda create -n cowbytes python=3.8 -y
conda activate cowbytes
```
Make sure to run ```conda activate cowbytes``` before running any of the scripts in this repo.

### Installing Dependencies
For non-Mac OS, install PyTorch by running the following:
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
If you are on Mac OS with MPS acceleration, run the following instead:
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

**Note:** To fully utilise cuda, make sure to have nvidia graphics drivers installed and running. To check, run ```nvidia-smi```.


Clone the repo if not done already and go inside the repo:
```
git clone --recurse https://github.com/sttaseen/scrambmix.git
cd scrambmix
````
To install all the other modules, navigate to the root directory of this repo after cloning and run the following:
```
pip install -r requirements.txt
```

This one is optional but to use the conda environment in Notebook, run:
```
conda install ipykernel -y
ipython kernel install --user --name=cowbytes
```

## Setup
### Downloading and extracting the dataset
In order to download the dataset, an existing [kaggle token](https://www.kaggle.com/docs/api#:~:text=From%20the%20site%20header%2C%20click,Create%20New%20API%20Token%E2%80%9D%20button.) needs to be set up.
All the data-acquisition and extraction is handled by ```setup.sh```. From the ```setup``` directory of the repo, run one of the files in the following format:
```
bash setup.sh
```

**Note:** If on any other operating system than Linux/Mac, open the bash file and run each command one by one.

This bash script will download the dataset from kaggle (*Kaggle token needs to be set up for this*), extract and store the dataset under the ```data``` directory.


## Citations
```
@article{Mathisetal2018,
    title = {DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
    author = {Alexander Mathis and Pranav Mamidanna and Kevin M. Cury and Taiga Abe  and Venkatesh N. Murthy and Mackenzie W. Mathis and Matthias Bethge},
    journal = {Nature Neuroscience},
    year = {2018},
    url = {https://www.nature.com/articles/s41593-018-0209-y}}

 @article{NathMathisetal2019,
    title = {Using DeepLabCut for 3D markerless pose estimation across species and behaviors},
    author = {Nath*, Tanmay and Mathis*, Alexander and Chen, An Chi and Patel, Amir and Bethge, Matthias and Mathis, Mackenzie W},
    journal = {Nature Protocols},
    year = {2019},
    url = {https://doi.org/10.1038/s41596-019-0176-0}}
    
@InProceedings{Mathis_2021_WACV,
    author    = {Mathis, Alexander and Biasi, Thomas and Schneider, Steffen and Yuksekgonul, Mert and Rogers, Byron and Bethge, Matthias and Mathis, Mackenzie W.},
    title     = {Pretraining Boosts Out-of-Domain Robustness for Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {1859-1868}}
    
@article{Lauer2022MultianimalPE,
    title={Multi-animal pose estimation, identification and tracking with DeepLabCut},
    author={Jessy Lauer and Mu Zhou and Shaokai Ye and William Menegas and Steffen Schneider and Tanmay Nath and Mohammed Mostafizur Rahman and     Valentina Di Santo and Daniel Soberanes and Guoping Feng and Venkatesh N. Murthy and George Lauder and Catherine Dulac and M. Mathis and Alexander Mathis},
    journal={Nature Methods},
    year={2022},
    volume={19},
    pages={496 - 504}}

@article{insafutdinov2016eccv,
    title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
    author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele},
    booktitle = {ECCV'16},
    url = {http://arxiv.org/abs/1605.03170}}
```
