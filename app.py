import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# load the pre-trained model
@st.cache_resource  # cache the model
def load_model():
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
    return resnet18_model

# Load the model
resnet18_model = load_model()

# Define the transformation function
data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Define the class labels
class_labels = {
    0: 'adenocarcinoma_left.lower.lobe',
    1: 'large.cell.carcinoma_left.hilum',
    2: 'normal',
    3: 'squamous.cell.carcinoma_left.hilum'
}

st.title("Lung Cancer Detection from CT Scan")
st.write("Upload a lung scan image, and the model will predict the type.")

# File uploader
uploaded_file = st.file_uploader("Choose a lung scan image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Lung Scan", use_column_width=True)

    # Preprocess the image
    input_tensor = data_transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = resnet18_model(input_batch)
        probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()  # Get probabilities
        predicted_class = probabilities.argmax()
        predicted_label = class_labels[predicted_class]

    # Display prediction and probabilities
    st.write(f"**Prediction:** {predicted_label}")
    st.write("**Class Probabilities:**")
    for idx, prob in enumerate(probabilities):
        st.write(f"{class_labels[idx]}: {prob * 100:.2f}%")