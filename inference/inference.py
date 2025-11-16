import os
from train.train import SimplePetClassifer
import torch
import torchvision.transforms as transforms

from utils_inference import preprocess_image, predict, visualize_predictions, get_predicted_label

def inference_image(image):
    # Load the trained model
    model_path = "../models/pretrained_classifier.pth"  # or "scratch_classifier.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePetClassifer()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    original_image, image_tensor = preprocess_image(image, transform)
    probabilities = predict(model, image_tensor, device)
    predicted_label = get_predicted_label(probabilities, {0: 'Cat', 1: 'Dog'})
    return predicted_label
