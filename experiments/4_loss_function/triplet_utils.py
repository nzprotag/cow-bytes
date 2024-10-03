import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import models
import torch.nn as nn

class TripletDataset(datasets.ImageFolder):
    """ This is the custom triplet dataset for the salient poses dataset. """
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.classes = os.listdir(folder_path)  # List of class folders
        self.data = []  # To store triplet samples
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        for class_label in self.classes:
            class_folder = os.path.join(folder_path, class_label)
            if os.path.isdir(class_folder):
                video_folders = os.listdir(class_folder)
                for video_folder in video_folders:
                    video_path = os.path.join(class_folder, video_folder)
                    if os.path.isdir(video_path):
                        frames = os.listdir(video_path)
                        self.data.append((class_label, video_folder, frames))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor_class, video_name, frames = self.data[index]
        
        # Randomly select an anchor frame
        anchor_frame = random.choice(frames)
        anchor_path = os.path.join(self.folder_path, anchor_class, video_name, anchor_frame)

        # Positive sample: randomly select another frame from the same class (not necessarily from the same video)
        positive_class = anchor_class
        positive_video_name, positive_frames = self._get_random_video_and_frames(positive_class, video_name)
        positive_frame = random.choice(positive_frames)
        positive_path = os.path.join(self.folder_path, positive_class, positive_video_name, positive_frame)

        # Negative sample: select a frame from the same video but from a different class
        other_class = random.choice([c for c in self.classes if c != anchor_class])
        negative_path = self._get_negative_frame_path(other_class, video_name)

        # Load images
        anchor_image = self.load_image(anchor_path)
        positive_image = self.load_image(positive_path)
        negative_image = self.load_image(negative_path)

        # Convert the class label to a tensor (you may use one-hot encoding or class index based on your needs)
        label = self.class_to_idx[anchor_class]  # Assuming labels are based on class indices

        return anchor_image, positive_image, negative_image, label

    def _get_random_video_and_frames(self, class_label, excluded_video_name):
        """Select a random video from the specified class excluding the given video name."""
        class_folder = os.path.join(self.folder_path, class_label)
        video_folders = [v for v in os.listdir(class_folder) if v != excluded_video_name]
        selected_video = random.choice(video_folders)
        selected_video_path = os.path.join(class_folder, selected_video)
        frames = os.listdir(selected_video_path)
        return selected_video, frames

    def _get_negative_frame_path(self, other_class, video_name):
        """Get a frame from the same video name in the other class, or a random one if not found."""
        other_class_folder = os.path.join(self.folder_path, other_class)
        negative_video_path = os.path.join(other_class_folder, video_name)

        if os.path.isdir(negative_video_path):  # Check if the folder exists
            negative_frames = os.listdir(negative_video_path)
        else:
            # Select a random video if the same video name doesn't exist in the other class
            negative_video_name, negative_frames = self._get_random_video_and_frames(other_class, video_name)
            negative_video_path = os.path.join(other_class_folder, negative_video_name)
            negative_frames = os.listdir(negative_video_path)

        negative_frame = random.choice(negative_frames)
        return os.path.join(negative_video_path, negative_frame)

    def load_image(self, path):
        image = Image.open(path).convert("RGB")  # Ensure image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image
    
class SimpleTripletDataset(datasets.ImageFolder):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.classes = os.listdir(folder_path)  # List of class folders
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        # Create a dictionary to store all images per class for easy access
        self.images_per_class = {class_name: [] for class_name in self.classes}

        # Populate the dictionary with images
        for class_label in self.classes:
            class_folder = os.path.join(folder_path, class_label)
            if os.path.isdir(class_folder):
                # Gather all image paths
                for root, _, files in os.walk(class_folder):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add other formats if necessary
                            self.images_per_class[class_label].append(os.path.join(root, file))

    def __len__(self):
        return sum(len(images) for images in self.images_per_class.values())

    def __getitem__(self, index):
        # Get all images and their corresponding classes
        all_images = [(image, class_name) for class_name, images in self.images_per_class.items() for image in images]

        # Randomly select an anchor image and its class
        anchor_image_path, anchor_class = random.choice(all_images)
        
        # Load the anchor image
        anchor_image = self.load_image(anchor_image_path)

        # Positive sample: Randomly select another image from the same class
        positive_image_path = random.choice(self.images_per_class[anchor_class])
        positive_image = self.load_image(positive_image_path)

        # Negative sample: Randomly select an image from a different class
        negative_class = random.choice([c for c in self.classes if c != anchor_class])
        negative_image_path = random.choice(self.images_per_class[negative_class])
        negative_image = self.load_image(negative_image_path)

        # Convert the class label to a tensor (using class index)
        label = self.class_to_idx[anchor_class]  # Assuming labels are based on class indices

        return anchor_image, positive_image, negative_image, label

    def load_image(self, path):
        image = Image.open(path).convert("RGB")  # Ensure image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image
    

# 4. Define the model
class TripletModel(nn.Module):
    def __init__(self):
        super(TripletModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 128)  # Adjust output size

    def forward(self, x):
        return self.base_model(x)
    

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train_and_evaluate_triplet(model, train_loader, val_loader, triplet_loss, optimizer, num_epochs, patience, device, ckpt_name=None):
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Lists to store training and validation metrics
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()  # Set the model to training mode
        running_train_loss = 0.0

        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            # Compute triplet loss
            loss = triplet_loss(anchor_output, positive_output, negative_output)
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_train_loss += loss.item()

        # Compute average training loss
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0

        with torch.no_grad():  # No need to calculate gradients during validation
            for val_anchor, val_positive, val_negative in val_loader:
                val_anchor, val_positive, val_negative = val_anchor.to(device), val_positive.to(device), val_negative.to(device)

                # Forward pass
                val_anchor_output = model(val_anchor)
                val_positive_output = model(val_positive)
                val_negative_output = model(val_negative)

                # Compute triplet loss
                val_loss = triplet_loss(val_anchor_output, val_positive_output, val_negative_output)

                running_val_loss += val_loss.item()

        # Compute average validation loss
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving model.")
            if ckpt_name:
                torch.save(model.state_dict(), f'{ckpt_name}.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break  # Stop training if no improvement for 'patience' epochs

    return best_val_loss, train_losses, val_losses