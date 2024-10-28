## Demo

### Libraries
For this demo to work, please install `streamlit` and `ffmpeg`.

```
pip install streamlit
```

Instuctions on how to install `ffmpeg` can be found [here](https://www.ffmpeg.org/download.html).


### Setup

Download the ResNet checkpoints from [this link](https://mega.nz/folder/wwJC1TzS#Qh2GpZtpMNdmTUhSq7tBJw) and if the BiteCount dataset is not already downloaded, it can be downloaded from [here](https://www.kaggle.com/datasets/sttaseen/bitecount).

In the `streamlitapp.py` file, modify `CHECKPOINT_DIR` to point to the ResNet checkpoints folder and `ROOT_DIR` to set where the dataset is stored for the app to work.

### Running
To run the app, execute the following command:

```
streamlit run streamlitapp.py
```

The demo should look like this:


![demo](https://github.com/user-attachments/assets/ecfd2d31-370f-445f-b4c8-796a52092710)
