import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet18,ResNet18_Weights


def load_model(model_path):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

def classify_image(model, image_path):
    input_tensor = preprocess(image_path)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

if __name__ == "__main__":
    model_path = "CNN_V1.pth" 
    image_path = "showcase_images/ai_2.png"
    model = load_model(model_path)

    class_idx = classify_image(model, image_path)
    confidences = torch.nn.functional.softmax(model(preprocess(image_path)), dim=1)[0]
    print(f"Confidence for real: {confidences[0]:.4f}")
    print(f"Confidence for AI-generated: {confidences[1]:.4f}")
    if class_idx == 0:
        print("Prediction: Real")
    else:
        print("Prediction: AI-generated")

    print (f"true: {image_path}")