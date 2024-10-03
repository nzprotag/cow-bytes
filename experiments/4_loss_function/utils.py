import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import numpy as np
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if validation accuracy improves
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                # Early stopping check
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if early_stopping.early_stop:
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model




# Function to display an image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Function to evaluate the best model and display misclassified images
def evaluate_model(model, criterion, val_loader, class_names):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    running_corrects = 0
    misclassified_images = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Collect misclassified images
            misclassified = preds != labels.data
            for i in range(len(inputs)):
                if misclassified[i]:
                    misclassified_images.append((inputs[i].cpu(), labels[i].cpu(), preds[i].cpu()))

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)

    print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Display misclassified images
    if misclassified_images:
        print(f"\nDisplaying {len(misclassified_images)} misclassified images...\n")
        for i, (img, true_label, predicted_label) in enumerate(misclassified_images[:5]):  # Display up to 5 images
            plt.figure()
            imshow(img, title=f'True: {class_names[true_label]}, Pred: {class_names[predicted_label]}')
            plt.show()

def show_images(images, labels, class_names, num_images_to_show=16):
    """Displays up to 16 images in a grid along with their labels."""
    num_images = min(len(images), num_images_to_show)  # Limit to 16 images
    plt.figure(figsize=(12, 8))
    
    for i in range(num_images):
        ax = plt.subplot(4, 4, i + 1)  # Create a 4x4 grid
        img = images[i].numpy().transpose((1, 2, 0))  # Convert tensor to numpy array
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean  # Denormalize
        img = np.clip(img, 0, 1)  # Clip values to be between 0 and 1
        plt.imshow(img)
        plt.title(class_names[labels[i]])  # Show the label
        plt.axis('off')  # Turn off axis
        
    plt.tight_layout()
    plt.show()


import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device, ckpt_name=None):
    # Variables to track the best validation accuracy and the number of epochs without improvement
    best_val_acc = 0.0
    best_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass

            # Compute loss
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

            # Apply sigmoid to outputs and threshold to get predictions
            preds = torch.sigmoid(outputs) > 0.5  # Get binary predictions
            correct += preds.eq(labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total examples

        # Compute average training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_predictions = []
        all_val_probs = []  # To store predicted probabilities for ROC AUC

        with torch.no_grad():  # No need to calculate gradients during validation
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item()
                preds = torch.sigmoid(val_outputs) > 0.5  # Get binary predictions
                val_correct += preds.eq(val_labels).sum().item()  # Count correct predictions
                val_total += val_labels.size(0)  # Count total examples

                # Store labels and predictions for F1 score calculation
                all_val_labels.extend(val_labels.cpu().numpy())
                all_val_predictions.extend(preds.cpu().numpy())
                all_val_probs.extend(torch.sigmoid(val_outputs).cpu().numpy())  # Store predicted probabilities

        # Compute average validation loss and accuracy
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Compute F1 score
        f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')

        # Compute Precision and Recall
        precision = precision_score(all_val_labels, all_val_predictions)
        recall = recall_score(all_val_labels, all_val_predictions)

        # Compute ROC AUC
        roc_auc = roc_auc_score(all_val_labels, all_val_probs)

        print(f'Epoch [{epoch + 1}/{num_epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {f1:.4f}, '
              f'Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val ROC AUC: {roc_auc:.4f}')

        # Early stopping
        if val_acc > best_val_acc or f1 > best_f1:
            best_val_acc = val_acc
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_roc_auc = roc_auc
            epochs_without_improvement = 0  # Reset counter if the model improved
            print(f"Validation accuracy improved to {best_val_acc:.4f}. Saving model.")
            if ckpt_name:
                # You can save the model here if you want
                torch.save(model.state_dict(), f'{ckpt_name}.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break  # Stop training if no improvement for 'patience' epochs

    return best_val_acc, best_f1, best_precision, best_recall, best_roc_auc # Return additional metrics