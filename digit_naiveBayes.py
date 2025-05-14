''' Naive Bayes with incremented training and Laplace smoothing '''
import math
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

num_features = 784  # 28 x 28
num_feature_vals = 3

def load_dataset(digit_file, label_file, fraction=1):
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

    if fraction != 1:
        return x[:cutoff], y[:cutoff], len(y[:cutoff])
    return x[:cutoff], y[:cutoff]

validation_digit_images, validation_digit_labels = load_dataset("digitdata/validationimages", "digitdata/validationlabels")
test_digit_images, test_digit_labels = load_dataset("digitdata/testimages", "digitdata/testlabels")

start = 0.1
end = 1
step = 0.1
x = start
training_accuracies = []
outfile_name = "results/digit_naiveBayes_results.txt"
print(f"Writing to: {outfile_name}")
with open(outfile_name, "w") as outfile:
    while x < end:
        start_time = time.time()
        training_digit_images, training_digit_labels, number_digits = load_dataset("digitdata/trainingimages", "digitdata/traininglabels", x)
        # print(len(training_digit_images), len(training_digit_labels))

        # Train Naive Bayes
        class_counts = Counter(training_digit_labels)
        total_samples = len(training_digit_labels)

        # Prior probabilities
        priors = {label: class_counts[label] / total_samples for label in class_counts}

        # Likelihoods: P(feature_i = value | class)
        # Structure: likelihoods[class][feature_index][feature_value]
        likelihoods = defaultdict(lambda: [Counter() for _ in range(num_features)])

        for features, label in zip(training_digit_images, training_digit_labels):
            for i, val in enumerate(features):
                likelihoods[label][i][val] += 1

        # Convert counts to probabilities with Laplace smoothing
        for label in likelihoods:
            for i in range(num_features):
                total = sum(likelihoods[label][i].values()) + num_feature_vals
                for val in range(num_feature_vals):
                    likelihoods[label][i][val] = (likelihoods[label][i][val] + 1) / total

        # Predition
        def predict(features):
            log_probs = {}
            for label in priors:
                log_prob = math.log(priors[label])
                for i, val in enumerate(features):
                    log_prob += math.log(likelihoods[label][i][val])
                log_probs[label] = log_prob
            return max(log_probs, key=log_probs.get)

        # Evaluate
        correct = 0
        for num, (features, label) in enumerate(zip(test_digit_images, test_digit_labels)):
            prediction = predict(features)
            if prediction == label:
                correct += 1
            # else:
            #     print(f"[{num}] Misclassified: Predicted {prediction}, Actual {label}")

        accuracy = correct / len(test_digit_labels)
        print(f"Percentage of Training Data used: {round(x * 100)}% ({number_digits} digits)", file=outfile)
        print(f"\tAccuracy: {accuracy:.2%}", file=outfile)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\tTraining time: {training_time:.2f} seconds\n", file=outfile)

        training_accuracies.append(accuracy * 100)
        # print(training_accuracies)

        x += step

print("Done!")
training_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plt.figure(figsize=(8, 4))
plt.plot(training_percents, training_accuracies, marker='o', color="maroon")

plt.title("Accuracy vs. Training Data Used (Digit Images)")
plt.xlabel("Training Data Used (%)")
plt.ylabel("Accuracy (%)")

plt.grid(True)
plt.xticks(training_percents)  # Ensure x-axis ticks match your data
# plt.ylim(0, 100)

plt.show()