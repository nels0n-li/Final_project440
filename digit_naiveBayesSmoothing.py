''' Naive Bayes with validation for better smoothing '''
import math
import time
from collections import defaultdict, Counter

num_features = 784  # 28 x 28
num_feature_vals = 3

def load_dataset(digit_file, label_file, fraction=1.0):
    def encode_digit(lines):
        char_map = {' ': 0, '+': 1, '#': 2}
        return [char_map[char] for line in lines for char in line.rstrip('\n')]

    with open(digit_file, 'r') as f:
        lines = f.readlines()
    x = [encode_digit(lines[i:i+28]) for i in range(0, len(lines), 28)]

    with open(label_file, 'r') as f:
        y = [int(line.strip()) for line in f]

    assert len(x) == len(y), f"Mismatched input: {len(x)} images vs {len(y)} labels"

    cutoff = int(min(len(x), len(y)) * fraction)
    return x[:cutoff], y[:cutoff]

# Train Naive Bayes
def train_naive_bayes(images, labels, k):
    class_counts = Counter(labels)
    total_samples = len(labels)
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
                count = likelihoods[label][i][val]
                smoothed_likelihoods[label][i][val] = (count + k) / total

    return priors, smoothed_likelihoods

# Predition
def predict(priors, likelihoods, features):
    log_probs = {}
    for label in priors:
        log_prob = math.log(priors[label])
        for i, val in enumerate(features):
            log_prob += math.log(likelihoods[label][i][val])
        log_probs[label] = log_prob
        
    return max(log_probs, key=log_probs.get)

# Evaluate
def evaluate(priors, likelihoods, images, labels):
    correct = 0
    for features, label in zip(images, labels):
        prediction = predict(priors, likelihoods, features)
        if prediction == label:
            correct += 1

    return correct / len(labels)


training_digit_images, training_digit_labels = load_dataset("digitdata/trainingimages", "digitdata/traininglabels")
validation_digit_images, validation_digit_labels = load_dataset("digitdata/validationimages", "digitdata/validationlabels")
test_digit_images, test_digit_labels = load_dataset("digitdata/testimages", "digitdata/testlabels")

k_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
best_k = None
best_validation_accuracy = 0

outfile_name = "results/digit_naiveBayesSmoothing_results.txt"
print(f"Writing to: {outfile_name}")
with open(outfile_name, "w") as outfile:
    for k in k_values:
        print(f"Training with k = {k}", file=outfile)
        start = time.time()
        priors, likelihoods = train_naive_bayes(training_digit_images, training_digit_labels, k)
        train_time = time.time() - start

        validation_accuracy = evaluate(priors, likelihoods, validation_digit_images, validation_digit_labels)
        print(f"\tValidation Accuracy: {validation_accuracy:.2%}", file=outfile)
        print(f"\tTraining Time: {train_time:.2f} seconds\n", file=outfile)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_k = k

    print(f"Best k = {best_k} with Validation Accuracy = {best_validation_accuracy:.2%}", file=outfile)

    priors, likelihoods = train_naive_bayes(training_digit_images, training_digit_labels, best_k)
    test_accuracy = evaluate(priors, likelihoods, test_digit_images, test_digit_labels)
    print(f"Test Accuracy with k = {best_k}: {test_accuracy:.2%}", file=outfile)

print("Done!")