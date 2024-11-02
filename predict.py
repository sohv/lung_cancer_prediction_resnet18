import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Load the model
resnet18_model = models.resnet18()
resnet18_model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.Linear(256, 4)
)
resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet18_model.load_state_dict(torch.load("chest-ctscan_model.pth"))
resnet18_model.eval()


preprocess = transforms.Compose([
    transforms.Resize(size=(224, 224)),
     transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_directory = "data/test/large.cell.carcinoma"

# list to store predictions
predictions = []

for filename in os.listdir(test_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(test_directory, filename)
        input_image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # Create a batch dimension

        # make prediction
        with torch.no_grad():
            output = resnet18_model(input_batch)
            _, predicted_class = torch.max(output, 1)  # get the predicted class index

        predictions.append((filename, predicted_class.item()))

# Print all predictions
for filename, class_idx in predictions:
    print(f"Image: {filename}, Predicted Class: {class_idx}")