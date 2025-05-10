import math
import time
from collections import defaultdict, Counter

num_features = 70*60
num_feature_vals = 2

def load_faces(face_file, label_file, fraction=1):
    char_map = {' ': 0, '#': 1}

    with open(face_file) as f:
        lines = f.readlines()

    x = [
        [char_map[char] for line in lines[i:i+70] for char in line.rstrip('\n')] for i in range(0, len(lines), 70)
    ]

    with open(label_file) as f:
        y = [int(line.strip()) for line in f]
        
    cutoff = int(len(x) * fraction)

    if fraction != 1:
        return x[:cutoff], y[:cutoff], len(y[:cutoff])
    return x[:cutoff], y[:cutoff]

def train_naive_bayes(images, labels, k):
    class_counts = Counter(labels)
    total_samples = len(images)
    priors = {label: class_counts[label] / total_samples for label in class_counts}

    likelihoods = defaultdict(lambda: [Counter() for _ in range(num_features)])

    for features, label in zip(images, labels):
        for i, val in enumerate(features):
            likelihoods[label][i][val] += 1

    smoothed_likelihoods = defaultdict(lambda: [{} for _ in range(num_features)])
    for label in likelihoods:
        for i in range(num_features):
            total = sum(likelihoods[label][i].values()) + k * num_feature_vals
            for val in range(num_feature_vals):
                smoothed_likelihoods[label][i][val] = (likelihoods[label][i][val] + k) / total

    return priors, smoothed_likelihoods

def predict(priors, likelihoods, features):
    log_probs = {}
    for label in priors:
        log_prob = math.log(priors[label])
        for i, val in enumerate(features):
            log_prob += math.log(likelihoods[label][i][val])
        log_probs[label] = log_prob
    
    return max(log_probs, key=log_probs.get)
    

def evaluate(priors, likelihoods, images, labels):
    correct = 0
    for features, label in zip(images, labels):
        prediction = predict(priors, likelihoods, features)
        if prediction == label:
            correct += 1

    return correct / len(labels)

training_face_images, training_face_labels = load_faces("facedata/facedatatrain", "facedata/facedatatrainlabels")
validation_face_images, validation_face_labels = load_faces("facedata/facedatavalidation", "facedata/facedatavalidationlabels")
test_face_images, test_face_labels = load_faces("facedata/facedatatest", "facedata/facedatatestlabels")

k_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
best_k = None
best_validation_accuracy = 0

outfile_name = "results/face_naiveBayesSmoothing_results.txt"
print(f"Writing to: {outfile_name}")
with open(outfile_name, "w") as outfile:
    for k in k_values:
        print(f"Training with k = {k}", file=outfile)
        start_time = time.time()
        priors, likelihoods = train_naive_bayes(training_face_images, training_face_labels, k)
        train_time = time.time() - start_time

        validation_accuracy = evaluate(priors, likelihoods, validation_face_images, validation_face_labels)
        print(f"\tValidation Accuracy: {validation_accuracy:.2%}", file=outfile)
        print(f"\tTraining Time: {train_time:.2f} seconds\n", file=outfile)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_k = k
        
    print(f"Best k = {best_k} with Validation Accuracy = {best_validation_accuracy:.2%}", file=outfile)

    priors, likelihoods = train_naive_bayes(training_face_images, training_face_labels, best_k)
    test_accuracy = evaluate(priors, likelihoods, test_face_images, test_face_labels)
    print(f"Test Accuracy with k = {best_k}: {test_accuracy:.2%}", file=outfile)

print("Done!")