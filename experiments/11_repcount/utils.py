import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch  

# Define the transformation pipeline
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

# Function to extract frames and apply transformations, output in TCHW format
def process_video_frames_tchw(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        
        # Convert the frame from BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame into a PIL image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply the transformations
        transformed_frame = test_transforms(pil_image)
        
        # Append the transformed frame to the list
        frame_list.append(transformed_frame)
    
    # Release the video capture object
    cap.release()
    
    # Stack the list of frames into a tensor of shape (T, C, H, W)
    tchw_tensor = torch.stack(frame_list)
    
    return tchw_tensor 

import numpy as np
import torch

class Action_trigger(object):
    """
        Trigger the salient action 1 or 2 during inference.
        This is used to calculate the repetitive count.
    """
    def __init__(self, action_name, enter_threshold=0.8, exit_threshold=0.4):
        self._action_name = action_name

        # If the score larger than the given enter_threshold, then that pose will enter the triggering.
        # If the score smaller than the given exit_threshold, then that pose will complete the triggering.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # Whether the pose has entered the triggering.
        self._pose_entered = False

    def __call__(self, pose_score):
        # We use two thresholds.
        # First, you need to enter the pose from a higher position above,
        # and then you need to exit from a lower position below.
        # The difference between the thresholds makes it stable against prediction jitter
        # (which would lead to false counts if there was only one threshold).

        triggered = False

        # On the very first frame or if we were out of the pose,
        # just check if we entered it on this frame and update the state.
        if not self._pose_entered:
            self._pose_entered = pose_score > self._enter_threshold
            return triggered

        # If we are in a pose and are exiting it, update the state.
        if pose_score < self._exit_threshold:
            self._pose_entered = False
            triggered = True

        return triggered
    
def count_predictions(y_hat, gt_count, enter_threshold=0.50, exit_threshold=0.50, momentum=0.4):
    """
    Y: tensor or numpy array of model predictions
    config: configuration dictionary containing Action_trigger thresholds
    index2action: dictionary mapping index to action names
    gt_count: ground truth count for comparison
    """

    # Initialize counters for the two salient actions
    repetition_salient_1 = Action_trigger(
        action_name=None,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold)
    repetition_salient_2 = Action_trigger(
        action_name=None,
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold)

    classify_prob = 0.5
    pose_count = 0
    curr_pose = 'holder'
    init_pose = 'pose_holder'
    classify_probs = []

    # Loop through the predictions to track pose counts
    for output_numpy in y_hat:
        classify_prob = output_numpy * (1. - momentum) + momentum * classify_prob
        classify_probs.append(classify_prob)

        # Trigger detection
        salient1_triggered = repetition_salient_1(classify_prob)
        reverse_classify_prob = 1 - classify_prob
        salient2_triggered = repetition_salient_2(reverse_classify_prob)

        if init_pose == 'pose_holder':
            if salient1_triggered:
                init_pose = 'salient1'
            elif salient2_triggered:
                init_pose = 'salient2'

        if init_pose == 'salient1':
            if curr_pose == 'salient1' and salient2_triggered:
                pose_count += 1
        else:
            if curr_pose == 'salient2' and salient1_triggered:
                pose_count += 1

        # Update current pose
        if salient1_triggered:
            curr_pose = 'salient1'
        elif salient2_triggered:
            curr_pose = 'salient2'

    # Calculate MAE
    mae = abs(gt_count - pose_count) / (gt_count + 1e-9)

    # Calculate OBO: One-off-by-one condition
    obo = 1 if abs(gt_count - pose_count) <= 1 else 0

    return pose_count, mae, obo