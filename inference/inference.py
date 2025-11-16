import os
from train.train import SimplePetClassifer
import torch
import torchvision.transforms as transforms
import numpy as np

from .utils_inference import preprocess_image, predict, visualize_predictions, get_predicted_label

def inference_image(image_path):
    assert os.path.isfile(image_path), f"Provided path is not a file: {image_path}"
    assert os.path.exists(image_path), f"Image file not found at {image_path}"
    
    # Load the trained model
    model_path = "models/pretrained_classifier.pth"  # or "scratch_classifier.pth" or "pretrained_classifier.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePetClassifer()
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    original_image, image_tensor = preprocess_image(image_path, transform)
    probabilities = predict(model, image_tensor, device)
    predicted_label = get_predicted_label(probabilities, {0: 'Cat', 1: 'Dog'})
    return predicted_label, original_image, probabilities