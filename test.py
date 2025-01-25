import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
X_data = pd.read_csv(r'C:\Users\KIIT\Downloads\logisticX.csv', header=None)
Y_data = pd.read_csv(r'C:\Users\KIIT\Downloads\logisticY.csv', header=None)

# Normalize the independent variables
X_normalized = (X_data - X_data.mean()) / X_data.std()

# Add a column of ones for the intercept term
X_normalized.insert(0, 'Intercept', 1)

# Convert to numpy arrays
X = X_normalized.values
Y = Y_data.values

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cost function
def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = sigmoid(X @ theta)
    cost = (-1 / m) * (Y.T @ np.log(predictions) + (1 - Y).T @ np.log(1 - predictions))
    return cost[0][0]

# Gradient descent implementation
def gradient_descent(X, Y, theta, learning_rate, iterations):
    m = len(Y)
    cost_history = []

    for _ in range(iterations):
        predictions = sigmoid(X @ theta)
        theta -= (learning_rate / m) * (X.T @ (predictions - Y))
        cost_history.append(compute_cost(X, Y, theta))

    return theta, cost_history

# Initialize variables
learning_rate = 0.1
iterations = 1000
theta_initial = np.zeros((X.shape[1], 1))

# Train logistic regression model
theta_optimal, cost_history = gradient_descent(X, Y, theta_initial, learning_rate, iterations)
final_cost = compute_cost(X, Y, theta_optimal)

# Plot cost function vs iterations
plt.figure(figsize=(8, 6))
plt.plot(range(1, iterations + 1), cost_history, label="Learning Rate: 0.1")
plt.title("Cost Function vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost Function Value")
plt.legend()
plt.grid()
plt.show()

# Plot dataset with decision boundary
plt.figure(figsize=(8, 6))
class_0 = Y.flatten() == 0
class_1 = Y.flatten() == 1
plt.plot(X[class_0, 1], X[class_0, 2], 'bo-', label='Class 0')
plt.plot(X[class_1, 1], X[class_1, 2], 'ro-', label='Class 1')

# Plot decision boundary
x_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
y_values = -(theta_optimal[0] + theta_optimal[1] * x_values) / theta_optimal[2]
plt.plot(x_values, y_values, 'g-', label='Decision Boundary')

plt.title("Dataset with Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()

# Train model with two learning rates and plot cost function
learning_rates = [0.1, 5]
plt.figure(figsize=(8, 6))

for lr in learning_rates:
    theta_initial = np.zeros((X.shape[1], 1))
    _, cost_history_lr = gradient_descent(X, Y, theta_initial, lr, 100)
    plt.plot(range(1, 101), cost_history_lr, label=f"Learning Rate: {lr}")

plt.title("Cost Function vs Iterations for Different Learning Rates")
plt.xlabel("Iterations")
plt.ylabel("Cost Function Value")
plt.legend()
plt.grid()
plt.show()

# Confusion matrix and metrics
def confusion_matrix_metrics(X, Y, theta):
    predictions = sigmoid(X @ theta) >= 0.5
    tp = np.sum((predictions == 1) & (Y == 1))
    tn = np.sum((predictions == 0) & (Y == 0))
    fp = np.sum((predictions == 1) & (Y == 0))
    fn = np.sum((predictions == 0) & (Y == 1))

    accuracy = (tp + tn) / len(Y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Confusion Matrix": [[tn, fp], [fn, tp]],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
    }

# Calculate metrics
metrics = confusion_matrix_metrics(X, Y, theta_optimal)

# Display metrics
metrics
