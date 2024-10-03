import os
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from triplet_utils import *
from utils import *
from torch.utils.data import random_split
from triplet_utils import train_and_evaluate_triplet

# TripletModel definition
class TripletModel(nn.Module):
    def __init__(self):
        super(TripletModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 128)  # Adjust output size for embedding

    def forward(self, x):
        # Get the embedding directly from the base model
        return self.base_model(x)  # Call the model with input x

class TripletClassifier(nn.Module):
    def __init__(self, triplet_head, output_size=1):
        super(TripletClassifier, self).__init__()
        self.triplet_head = triplet_head
        self.classifier = nn.Linear(self.triplet_head.base_model.fc.out_features, output_size)  # Binary classification (logits)

    def forward(self, x):
        embedding = self.triplet_head(x)  # Output of base model
        class_output = self.classifier(embedding)  # Output for classification
        return embedding, class_output
    

    # Load data
data_dir = '../../data/BiteCount/salient_poses/'

# Define transformations for training set and validation set
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing to square input
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing to square input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
dataset = TripletDataset(data_dir, transform=train_transforms)
dataset_size = len(dataset)
class_names = dataset.classes

dataset_size = len(dataset)
class_names = dataset.classes

# # Cross-validation setup
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Split the dataset randomly into training and validation sets
train_size = int(0.8 * dataset_size)  # 80% training, 20% validation
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Hyperparameters
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 25
patience = 5
alpha = 0.5

# Placeholder to store metrics across folds
val_accuracies = []
val_f1_scores = []
precisions = []
recalls = []
roc_aucs = []

def custom_collate(batch):
    # Unpack the batch
    anchors, positives, negatives, labels = zip(*batch)

    # Stack tensors to create batches
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return anchors, positives, negatives, labels


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

# Perform 5-Fold Cross Validation
fold_idx = 1
num_epochs = 25
patience = 5

import itertools
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch.optim as optim
import torch.nn as nn
import numpy as np

# Define the range of hyperparameters to test
margins = [i * 0.1 for i in range(1, 11)]  # Margin from 0.1 to 1.0
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different values for alpha
learning_rates = [0.001, 0.01, 0.1]  # Example learning rates
momentums = [0.5, 0.8, 0.9]  # Different momentum values

# Store the best results
best_val_acc = 0.0
best_hyperparams = {}
best_val_loss = 1e10

# Create a search space using itertools
search_space = itertools.product(margins, alphas, learning_rates, momentums)

for margin, alpha, lr, momentum in search_space:
    print(f"Training with margin: {margin}, alpha: {alpha}, learning_rate: {lr}, momentum: {momentum}")

    # Initialize TripletModel
    triplet_model = TripletModel()
    model = TripletClassifier(triplet_model)
    model = model.to(device)

    # Define loss functions and optimizer
    criterion_classification = nn.BCEWithLogitsLoss()  # Binary cross-entropy for classification
    criterion_triplet = nn.TripletMarginLoss(margin=margin)  # Use margin from the search space
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # Optimizer with dynamic lr and momentum
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            anchor, positive, negative, labels = batch
            anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)

            # Forward pass
            anchor_embedding, anchor_output = model(anchor)
            positive_embedding, positive_output = model(positive)
            negative_embedding, negative_output = model(negative)

            # Calculate losses
            classification_loss = criterion_classification(anchor_output, labels)
            triplet_loss = criterion_triplet(anchor_embedding, positive_embedding, negative_embedding)

            # Combine losses
            loss = alpha * classification_loss + (1 - alpha) * triplet_loss
            running_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                anchor, positive, negative, labels = batch
                anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)

                anchor_embedding, anchor_output = model(anchor)
                positive_embedding, positive_output = model(positive)
                negative_embedding, negative_output = model(negative)

                # Compute validation loss
                classification_loss = criterion_classification(anchor_output, labels)
                triplet_loss = criterion_triplet(anchor_embedding, positive_embedding, negative_embedding)
                val_loss += (classification_loss + triplet_loss).item()

                val_preds.append(torch.sigmoid(anchor_output).cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        # Calculate validation metrics
        val_acc = accuracy_score(val_labels, (val_preds > 0.5).astype(int))
        val_f1 = f1_score(val_labels, (val_preds > 0.5).astype(int))
        precision = precision_score(val_labels, (val_preds > 0.5).astype(int))
        recall = recall_score(val_labels, (val_preds > 0.5).astype(int))
        roc_auc = roc_auc_score(val_labels, val_preds)

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}")

        # Early stopping and best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_hyperparams = {'margin': margin, 'alpha': alpha, 'learning_rate': lr, 'momentum': momentum}
            patience_counter = 0
            torch.save(model.state_dict(), f'triplet_classifier_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# Save the results to a text file
with open('best_hyperparameters_tmm.txt', 'w') as f:
    f.write(f"Best validation loss: {best_val_loss:.4f}\n")
    f.write(f"Best accuracy: {best_val_acc:.4f}\n")
    f.write(f"Best hyperparameters:\n")
    f.write(f"  Margin: {best_hyperparams['margin']}\n")
    f.write(f"  Alpha: {best_hyperparams['alpha']}\n")
    f.write(f"  Learning Rate: {best_hyperparams['learning_rate']}\n")
    f.write(f"  Momentum: {best_hyperparams['momentum']}\n")

print("Results saved to 'best_hyperparameters_results.txt'")
