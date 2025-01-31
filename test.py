import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
features = pd.read_csv(r'C:\Users\KIIT\Downloads\logisticX.csv', header=None)
target = pd.read_csv(r'C:\Users\KIIT\Downloads\logisticY.csv', header=None)

# Standardize the feature set
features_standardized = (features - features.mean()) / features.std()

# Append bias term (intercept)
features_standardized.insert(0, 'Bias', 1)

# Convert to numpy arrays
X = features_standardized.values
y = target.values

# Activation function (sigmoid)
def sigmoid_function(value):
    return 1 / (1 + np.exp(-value))

# Cost function
def compute_loss(X, y, params):
    num_samples = len(y)
    predictions = sigmoid_function(X @ params)
    loss = (-1 / num_samples) * (y.T @ np.log(predictions) + (1 - y).T @ np.log(1 - predictions))
    return loss[0][0]

# Gradient Descent with Early Stopping
def gradient_optimization(X, y, params, alpha, max_iterations, tolerance=1e-6):
    num_samples = len(y)
    loss_history = []
    previous_loss = float('inf')
    
    for _ in range(max_iterations):
        predictions = sigmoid_function(X @ params)
        params -= (alpha / num_samples) * (X.T @ (predictions - y))
        current_loss = compute_loss(X, y, params)
        loss_history.append(current_loss)
        
        # Early stopping condition
        if abs(previous_loss - current_loss) < tolerance:
            break
        previous_loss = current_loss
    
    return params, loss_history

# Initialize parameters
learning_rate = 0.1
iterations = 1000
params_initial = np.zeros((X.shape[1], 1))

# Train the model
optimized_params, loss_values = gradient_optimization(X, y, params_initial, learning_rate, iterations)
final_loss = compute_loss(X, y, optimized_params)

# Plot loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(loss_values) + 1), loss_values, label="Learning Rate: 0.1")
plt.title("Loss Function vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss Value")
plt.legend()
plt.grid()
plt.show()

# Visualize dataset with decision boundary (without scatter plot)
plt.figure(figsize=(8, 6))
positive_class = y.flatten() == 1
negative_class = y.flatten() == 0
plt.plot(X[positive_class, 1], X[positive_class, 2], 'ro-', label='Positive Class')
plt.plot(X[negative_class, 1], X[negative_class, 2], 'bo-', label='Negative Class')

# Compute decision boundary
x_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
y_vals = -(optimized_params[0] + optimized_params[1] * x_vals) / optimized_params[2]
plt.plot(x_vals, y_vals, 'g-', label='Decision Boundary')

plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()

# Additional third graph: Cost function comparison with different learning rates
learning_rates = [0.1, 0.5]
plt.figure(figsize=(8, 6))
for lr in learning_rates:
    params_initial = np.zeros((X.shape[1], 1))
    _, loss_values_lr = gradient_optimization(X, y, params_initial, lr, 100)
    plt.plot(range(1, len(loss_values_lr) + 1), loss_values_lr, label=f"Learning Rate: {lr}")

plt.title("Cost Function vs Iterations for Different Learning Rates")
plt.xlabel("Iterations")
plt.ylabel("Loss Value")
plt.legend()
plt.grid()
plt.show()

# Model performance metrics
def performance_metrics(X, y, params):
    predictions = sigmoid_function(X @ params) >= 0.5
    true_positives = np.sum((predictions == 1) & (y == 1))
    true_negatives = np.sum((predictions == 0) & (y == 0))
    false_positives = np.sum((predictions == 1) & (y == 0))
    false_negatives = np.sum((predictions == 0) & (y == 1))
    
    accuracy = (true_positives + true_negatives) / len(y)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "Confusion Matrix": [[true_negatives, false_positives], [false_negatives, true_positives]],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
    }

# Compute and display performance metrics
metrics_output = performance_metrics(X, y, optimized_params)
print(metrics_output)
