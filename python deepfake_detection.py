import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

# 1. Dataset Class
class DeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label

# 2. Data Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load Data
data_dir = "path_to_dataset"
real_dir = os.path.join(data_dir, "real")
fake_dir = os.path.join(data_dir, "fake")

real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]

file_paths = real_files + fake_files
labels = [0] * len(real_files) + [1] * len(fake_files)

train_paths, val_paths, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

train_dataset = DeepfakeDataset(train_paths, train_labels, transform=transform)
val_dataset = DeepfakeDataset(val_paths, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Model Definition
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Adjust the final layer for binary classification
model = model.cuda()

# 5. Training Setup
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.cuda(), labels.float().unsqueeze(1).cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.float().unsqueeze(1).cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(loader), correct / total

# 6. Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = validate(model, val_loader, criterion)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

# 7. Save Model
torch.save(model.state_dict(), "deepfake_detector.pth")

# 8. Real-Time Video Processing
def detect_deepfake_in_video(video_path, model, transform):
    model.eval()
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = transform(rgb_frame).unsqueeze(0).cuda()

        # Model prediction
        with torch.no_grad():
            output = model(image)
            prediction = torch.sigmoid(output).item()

        # Annotate frame
        label = "Fake" if prediction > 0.5 else "Real"
        color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
        cv2.putText(frame, f"{label} ({prediction:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display frame
        cv2.imshow("Deepfake Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
detect_deepfake_in_video("path_to_video.mp4", model, transform)
