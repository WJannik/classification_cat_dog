import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
#from tqdm.notebook import tqdm
import timm
import matplotlib.pyplot as plt

class SimplePetClassifer(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(SimplePetClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(enet_out_size, num_classes) # or simply just this, since the output of EfficientNetB0 is 1280 and is already sufficient for binary classification
            nn.Linear(enet_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

def train_model(train_loader, test_loader, **kwargs):   
    """Train a classification model on the provided data loaders.
    Parameters:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        **kwargs: Additional configuration parameters.
    Returns:
        train_losses (list): List of training losses per epoch.
        test_losses (list): List of test losses per epoch.
        accuracies (list): List of accuracies per epoch.
    """
    # Default configuration
    defaults = {
        "num_epochs": 1,
        "learning_rate": 0.001,
        "data_augmentation": True,
        "pretrained": True,
        "model_name": "SimplePetClassifer",
        "use_LR_scheduler": False,
    }

    # Update defaults with any new values passed in kwargs
    config = {**defaults, **kwargs}

    # Set up metrics storage
    train_losses, test_losses, accuracies = [], [], []
    highest_accuracy = 0.0

    # Choose device and create model, loss function, optimizer and scheduler
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimplePetClassifer(num_classes=2, pretrained=config["pretrained"])
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    if config["use_LR_scheduler"]:
        scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    # Training loop
    for epoch in range(config["num_epochs"]):
        # Training phase
        print(f"------------------ Starting epoch {epoch+1} of {config['num_epochs']} ------------------")
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            # Add data augmentation if specified
            if config["data_augmentation"]:
                #1) flip horizontally
                if torch.rand(1) < 0.5:
                    images = torch.flip(images, dims=[3])
                #2) random crop
                if torch.rand(1) < 0.5:
                    crop_size = 64
                    start_x = torch.randint(0, images.size(2) - crop_size + 1, (1,)).item()
                    start_y = torch.randint(0, images.size(3) - crop_size + 1, (1,)).item()
                    images = images[:, :, start_x:start_x + crop_size, start_y:start_y + crop_size]
                    images = torch.nn.functional.interpolate(images, size=(128, 128), mode='bilinear', align_corners=False)

            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if config["use_LR_scheduler"]:
                scheduler.step()
            running_loss += loss.item() * labels.size(0)
        
        # Compute average training loss for the epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Testing phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            acumulated_correct = 0
            for images, labels in tqdm(test_loader, desc='Testing loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                # Compute accuracy based on highest probability
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                correct = (predicted == labels).sum().item()
                acumulated_correct += correct
            # Average accuracy for the epoch
            accuracy = acumulated_correct / len(test_loader.dataset)

        test_loss = running_loss / len(test_loader.dataset)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            # Saved model with highest accuracy
            print(f"New highest accuracy: {highest_accuracy}, saving model...")
            save_path = "../models/" + config["model_name"] + ".pth"
            torch.save(model.state_dict(), save_path)
        print(f"Train loss: {train_loss}, Test loss: {test_loss}, Accuracy: {accuracy}")
    return train_losses, test_losses, accuracies


def plot_metrics(train_losses, test_losses, accuracies):
    """Plot training and test losses and accuracy over epochs.
    Parameters:
        train_losses (list): List of training losses per epoch.
        test_losses (list): List of test losses per epoch.
        accuracies (list): List of accuracies per epoch.
    Returns:
        None
    """
    epochs = range(1, len(train_losses) + 1)
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs,train_losses, label='Training loss')
    plt.plot(epochs,test_losses, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.title("Loss over epochs")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.legend()
    plt.title("Accuracy over epochs")
    plt.figure(figsize=(15, 5))
    plt.show()

