import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_image(image_path: str, transform: torch.nn.Module) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert('RGB')
    return image, transform(image).unsqueeze(0)

# Predict using the model
def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# Get predicted label based on probabilities
def get_predicted_label(probabilities: np.ndarray, target_to_class: dict) -> str:
    predicted_index = np.argmax(probabilities)
    return target_to_class[predicted_index]

# Visualization
def visualize_predictions(original_image: Image.Image, probabilities: np.ndarray, class_names: list[str]) -> None:
    fig, axarr = plt.subplots(1, 2, figsize=(7, 4))
    
    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    
    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    predicted_class = class_names[np.argmax(probabilities)]
    plt.suptitle(f"Predicted: {predicted_class}", fontsize=16)
    plt.figuresize=(6,3)
    plt.show()