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



test_face_images, test_face_labels = load_faces("facedata/facedatatest", "facedata/facedatatestlabels")

start = 0.1
end = 1
step = 0.1
z = start
outfile_name = "results/face_naiveBayes_results.txt"
print(f"Writing to: {outfile_name}")
with open(outfile_name, "w") as outfile:
    while z < end:
        start_time = time.time()

        training_face_images, training_face_labels, number_images = load_faces("facedata/facedatatrain", "facedata/facedatatrainlabels", z)
        priors, likelihoods = train_naive_bayes(training_face_images, training_face_labels, 1)
        accuracy = evaluate(priors, likelihoods, test_face_images, test_face_labels)

        print(f"Percentage of Training Data used: {round(z * 100)}% ({number_images} images)", file=outfile)
        print(f"Accuracy: {accuracy:.2%}", file=outfile)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds\n", file=outfile)

        z += step

print("Done!")