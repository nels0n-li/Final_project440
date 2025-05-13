import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

train_image_path = r"digitdata\trainingimages"
train_label_path = r"digitdata\traininglabels"
test_image_path = r"digitdata\testimages"      
test_label_path = r"digitdata\testlabels"      

image_height, image_width = 28, 28
lines_per_image = image_height
char_to_pixel = {' ': 0, '+': 1, '#': 2}

def load_images(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    num_images = len(lines) // lines_per_image
    images = []
    for i in range(num_images):
        block = lines[i * lines_per_image : (i + 1) * lines_per_image]
        image = [[char_to_pixel.get(c, 0) for c in line.ljust(image_width)] for line in block]
        images.append(np.array(image).flatten())
    return np.array(images)


def load_labels(path):
    with open(path, "r") as f:
        return np.array([int(line.strip()) for line in f.readlines()])


X_train = load_images(train_image_path)
y_train = load_labels(train_label_path)

X_test = load_images(test_image_path)
y_test = load_labels(test_label_path)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


C_value = 10 # Regularization parameter
clf = SVC(kernel='linear', C=C_value)
clf.fit(X_train_scaled, y_train)


y_pred = clf.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Regularization parameter: {C_value}")
print(f"Test accuracy: {test_accuracy:.4f}")
