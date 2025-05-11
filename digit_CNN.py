import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

height = 28
width = 28
size = 32
epochs = 10
output = "results/digit_cnn_results.txt"

char_to_pixel = {' ': 0, '+': 1, '#': 2}

# Load ASCII Images
def load_images(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    num_images = len(lines) // height
    images = []
    for i in range(num_images):
        block = lines[i * height : (i + 1) * height]
        image = [[char_to_pixel.get(c, 0) for c in line.ljust(width)] for line in block]
        images.append(image)
    return np.array(images, dtype=np.float32)

def load_labels(path):
    with open(path, "r") as f:
        return np.array([int(line.strip()) for line in f.readlines()], dtype=np.int64)

#CNN Model
def build_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

#Training Loop
def train_model(model, train_loader, criterion, optimizer, device, file):
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

    duration = time.time() - start_time
    log = f"Loss: {total_loss:.4f}, Time: {duration:.2f} sec"
    print(log)
    print(log, file=file)
    return total_loss, duration

#Evaluation
def evaluate_model(model, test_loader, device, file):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    result = f"Test Accuracy: {accuracy:.4f}"
    print(result)
    print(result, file=file)
    return accuracy

#Main 
def run():
    X_train = load_images("digitdata/trainingimages")
    y_train = load_labels("digitdata/traininglabels")
    X_test = load_images("digitdata/testimages")
    y_test = load_labels("digitdata/testlabels")

    # Reshape and prepare tensors
    X_train_tensor = torch.tensor(X_train).unsqueeze(1) 
    y_train_tensor = torch.tensor(y_train)
    X_test_tensor = torch.tensor(X_test).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with open(output, "w") as f:
        total_start = time.time()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}", file=f)
            train_model(model, train_loader, criterion, optimizer, device, f)

        total_duration = time.time() - total_start
        print(f"Total Training Time: {total_duration:.2f} sec")
        print(f"Total Training Time: {total_duration:.2f} sec", file=f)

        evaluate_model(model, test_loader, device, f)

run()
