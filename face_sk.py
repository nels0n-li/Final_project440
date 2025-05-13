import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_faces(file_path, img_height=70, img_width=60):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    num_images = len(lines) // img_height
    images = []
    for i in range(num_images):
        block = lines[i * img_height : (i + 1) * img_height]
        img = [[1 if c != ' ' else 0 for c in line.ljust(img_width)] for line in block]
        images.append(np.array(img).flatten())
    return np.array(images)

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return np.array([int(line.strip()) for line in f.readlines()])

X_train = load_faces(r"facedata\facedatatrain")
y_train = load_labels(r"facedata\facedatatrainlabels")

X_test = load_faces(r"facedata\facedatatest")
y_test = load_labels(r"facedata\facedatatestlabels")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


C_value = 1e5
clf = SVC(kernel='linear', C=C_value)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Regularization parameter: {C_value}")
print(f"Test accuracy: {accuracy:.4f}")
