import streamlit as st
import os 
import imageio 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import os
from utils import process_video_frames_tchw, count_predictions
import numpy as np

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Root directory for dataset
ROOT_DIR = '../../data/BiteCount'
FOLDS_DIR = './folds/'  
VIDEO_DIR = os.path.join(ROOT_DIR, 'video')

# Checkpoint path
CHECKPOINT_DIR = '/media/sadat/sadat/resnet_benchmarks'

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

def predict_video(row, ):
    """Returns prediction, mae, and obo"""
    video_path = os.path.join(VIDEO_DIR, row['name'])
    ground_truth = row['count']
    transformed_tchw_tensor = process_video_frames_tchw(video_path).to(device)
    sigmoid = nn.Sigmoid()
    batch_size = BATCH_SIZE

    Y = []

    model.eval() 
    with torch.no_grad(): 
        for i in range(0, len(transformed_tchw_tensor)+1, batch_size):
            batch = transformed_tchw_tensor[i:i+batch_size].cuda() 
            output = model(batch)  
            y_batch = sigmoid(output).cpu().numpy()  
            Y.append(y_batch)  

    Y = np.concatenate(Y, axis=0)  
    Y = Y.squeeze()
    return Y, count_predictions(Y,
                             ground_truth,
                             ENTER_THRESHOLD,
                             EXIT_THRESHOLD,
                             MOMENTUM)

# Setup the sidebar
with st.sidebar: 
    st.image('https://i.pinimg.com/550x/51/45/66/5145668e8e638ae7341fa408a76a0fbf.jpg')
    st.title('CowBytes')
    st.info('This is a demo for our CowBytes model that encodes bite videos into a 1D signal and then counts the number of bites from the signal.')

    st.header('BiteCount Dataset')
    fold = st.selectbox('Select Fold Number', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    st.header('Classifier')
    resnet_model = st.selectbox('Select ResNet Model', options=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'])
    BATCH_SIZE = st.slider('Batch Size', 1, 64, 8, 1)

    st.header('Action Trigger Module')
    ENTER_THRESHOLD = st.slider('Enter Threshold', 0.0, 1.0, 0.78, 0.01)
    EXIT_THRESHOLD = st.slider('Exit Threshold', 0.0, 1.0, 0.4, 0.01)
    MOMENTUM = st.slider('Momentum', 0.0, 1.0, 0.4, 0.01)
    
st.title('Bite Counting Demo') 

# Initialize the selected ResNet model
if resnet_model == 'ResNet18':
    model = models.resnet18(pretrained=False)
elif resnet_model == 'ResNet34':
    model = models.resnet34(pretrained=False)
elif resnet_model == 'ResNet50':
    model = models.resnet50(pretrained=False)
elif resnet_model == 'ResNet101':
    model = models.resnet101(pretrained=False)
elif resnet_model == 'ResNet152':
    model = models.resnet152(pretrained=False)

# Initialize fc 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

# Load the corresponding checkpoint
checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_{resnet_model.lower()}_fold{fold}.pth')
model.load_state_dict(torch.load(checkpoint_path))

test_annotation_file = os.path.join(FOLDS_DIR, f'test_fold_{fold}.csv')
df = pd.read_csv(test_annotation_file)

# Generating a list of options or videos 
options = df['name'].unique()
selected_video = st.selectbox('Choose video', options)
row_number = df.index[df['name'] == selected_video][0]

# Generate two columns 
col1, col2 = st.columns(2)

if len(options) > 0: 

    # Rendering the video 
    with col1: 
        st.info('**Video Preview**')
        file_path = os.path.join(VIDEO_DIR, selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('**Model Predictions**')

        # Run the prediction for the selected video row
        Y, (pose_count, mae, obo) = predict_video(df.loc[row_number])
        ground_truth = df.loc[row_number, 'count']
        
        # Create the DataFrame for plotting
        frame_plot = pd.DataFrame({'Frame': range(len(Y)), 'Predictions': Y})

        # Create the Seaborn plot for predictions
        sns.set_style('whitegrid') 
        plt.figure(figsize=(9, 3)) 
        sns.lineplot(data=frame_plot, x='Frame', y='Predictions', color='b', marker='o', linewidth=2.5)

        # Add horizontal lines for the enter and exit thresholds
        plt.axhline(y=ENTER_THRESHOLD, color='g', linestyle='--')
        plt.axhline(y=EXIT_THRESHOLD, color='r', linestyle='--')

        # Customize the plot
        plt.xlabel('Frame Number', fontsize=14) 
        plt.ylabel('Prediction', fontsize=14)    
        plt.xticks(fontsize=12)                   
        plt.yticks(fontsize=12)                  

        # Render the plot in Streamlit
        st.pyplot(plt, use_container_width=True)

        st.text(f"Ground Truth Count: {ground_truth}")
        st.text(f"Predicted Count: {pose_count}")
        st.text(f"Off-By-One Accuracy (OBOA): {obo}")
        st.text(f"Mean Absolute Error (MAE): {mae:.2f}")