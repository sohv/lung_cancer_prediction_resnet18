import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torchvision.models as models
import torch.nn as nn

resnet18_model = models.resnet18()
resnet18_model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.Linear(256, 4)
)
resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet18_model.load_state_dict(torch.load("chest-ctscan_model (1).pth"))
resnet18_model.eval()

# Define your transformation function
data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
     transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_directory = "data/test/adenocarcinoma"
true_labels = []
predictions = []

def get_true_label(filename):
    label = 0
    return label

# iterate through test images
for filename in os.listdir(test_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(test_directory, filename)
        input_image = Image.open(image_path).convert("RGB")
        input_tensor = data_transform(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a batch dimension

        with torch.no_grad():
            output = resnet18_model(input_batch)
            _, predicted_class = torch.max(output, 1)  # get the predicted class index

        predictions.append(predicted_class.item())

        true_labels.append(get_true_label(filename))

# Calculate performance metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')
confusion_mat = confusion_matrix(true_labels, predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(confusion_mat)
