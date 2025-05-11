import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

height = 70
width = 60
batch_size = 32
epochs = 10
output = "results/face_cnn_results.txt"

#Load ASCII Images
def load_faces(file_path, height=70, width=60):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    num_images = len(lines) // height
    images = []
    for i in range(num_images):
        block = lines[i * height : (i + 1) * height]
        img = [[1 if c != ' ' else 0 for c in line.ljust(width)] for line in block]
        images.append(img)
    return np.array(images, dtype=np.float32)

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return np.array([int(line.strip()) for line in f], dtype=np.int64)

#Load Data
X_train = load_faces("facedata/facedatatrain")
y_train = load_labels("facedata/facedatatrainlabels")
X_test = load_faces("facedata/facedatatest")
y_test = load_labels("facedata/facedatatestlabels")
X_train_tensor = torch.tensor(X_train).unsqueeze(1) 
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test).unsqueeze(1)
y_test_tensor = torch.tensor(y_test)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(torch.unique(y_train_tensor))

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(64 * 17 * 15, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train and Log to File
with open(output, "w") as f:
    overall_start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        log = f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Time: {epoch_time:.2f} sec"
        print(log)
        print(log, file=f)

    total_time = time.time() - overall_start
    print(f"Total Training Time: {total_time:.2f} sec")
    print(f"Total Training Time: {total_time:.2f} sec", file=f)

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    result = f"Test Accuracy: {accuracy:.4f}"
    print(result)
    print(result, file=f)
