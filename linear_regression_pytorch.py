# A beginner-friendly implementation of a deep learning workflow using PyTorch.
# We build a simple linear model (the simplest form of a DNN) to predict a straight line.

import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# --- 0. Setup: Device Agnostic Code ---
# Set up device to use GPU if available, otherwise CPU. This ensures portability.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --- 1. Data: Create and Prepare Linear Data ---
# Create known parameters for our line equation (y = weight * X + bias)
WEIGHT = 0.7
BIAS = 0.3

# Create input data (X) and corresponding labels (y)
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) # dim=1 ensures correct shape for linear model
y = WEIGHT * X + BIAS

# Split data into training (80%) and testing (20%) sets
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Send data to target device (GPU/CPU)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


# --- 2. Build Model: The Neural Network Architecture ---
# Subclass nn.Module - this is the base class for all neural network modules in PyTorch.
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear is a deep learning "layer" that automatically creates 
        # the model's parameters (weights and biases) and performs the 
        # linear transformation (y = m*x + b).
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    # The forward method defines the computation that happens on the input data.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Create an instance of the model and move it to the target device
torch.manual_seed(42) # Set seed for reproducibility
model = LinearRegressionModelV2().to(device) 
print("\n--- Initial Model Parameters (Random) ---")
print(model.state_dict()) # Show the initial, random weights and bias


# --- 3. Loss Function and Optimizer ---
# The loss function measures how wrong the model's predictions are.
# For regression (predicting a number), Mean Absolute Error (MAE) is common.
loss_fn = nn.L1Loss() # Same as MAE

# The optimizer tells the model how to update its internal parameters 
# (weights and biases) to reduce the loss, using Gradient Descent.
optimizer = torch.optim.SGD(
    params=model.parameters(), # Parameters of the model to optimize
    lr=0.01                    # Learning Rate (controls the size of the updates)
)


# --- 4. Training Loop: The Learning Process ---
epochs = 1000
print(f"\n--- Training for {epochs} Epochs ---")

for epoch in range(epochs):
    
    # 1. Set model to training mode
    model.train()

    # 2. Forward pass: data goes through the model's forward() method
    y_pred = model(X_train)

    # 3. Calculate loss: compare prediction to ground truth
    loss = loss_fn(y_pred, y_train)

    # 4. Zero gradients: clear old gradients before calculating new ones
    optimizer.zero_grad()

    # 5. Backpropagation: calculate the gradient of the loss with respect 
    # to every parameter in the model (how much to adjust each parameter).
    loss.backward()

    # 6. Optimizer step: update the parameters based on the calculated gradients (Gradient Descent)
    optimizer.step()

    # --- Testing/Evaluation ---
    model.eval() # Set model to evaluation mode (turns off features like gradient tracking)
    with torch.inference_mode(): # Faster calculation when not training
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch:4d} | Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")

print("\n--- Final Model Parameters (Learned) ---")
print(model.state_dict())
print(f"Original Parameters: Weight={WEIGHT}, Bias={BIAS}")


# --- 5. Making Predictions (Inference) and Visualization ---
# Set model to evaluation mode
model.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model(X_test)

# Move data to CPU for use with Matplotlib
X_train_cpu, y_train_cpu = X_train.cpu().numpy(), y_train.cpu().numpy()
X_test_cpu, y_test_cpu = X_test.cpu().numpy(), y_test.cpu().numpy()
y_preds_cpu = y_preds.cpu().numpy()

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """Plots training data, test data and compares predictions."""
    plt.figure(figsize=(10, 7))
    plt.title("PyTorch Linear Regression Learning")
    plt.xlabel("X Feature")
    plt.ylabel("Y Label")
    
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data (True)")

    if predictions is not None:
        # Plot the predictions in red
        plt.scatter(test_data, predictions, c="r", s=8, marker='x', label="Predictions (Model Output)")

    plt.legend(prop={"size": 14})
    plt.grid(True)
    plt.show()

# Generate plot and save it
plot_predictions(X_train_cpu, y_train_cpu, X_test_cpu, y_test_cpu, y_preds_cpu)
plt.savefig("model_predictions.png")
print("\nPrediction plot generated and saved as 'model_predictions.png'")


# --- 6. Saving and Loading the Model ---
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True) # Create models directory if it doesn't exist

MODEL_NAME = "linear_regression_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model's state_dict (recommended method for inference)
print(f"\nSaving model state_dict to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

# Load the model state_dict
loaded_model = LinearRegressionModelV2().to(device)
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Verify predictions from the loaded model match the original model
loaded_model.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model(X_test)

# Check if the predictions are close (due to potential floating point differences)
is_close = torch.allclose(y_preds, loaded_model_preds)
print(f"Predictions from saved/loaded model match original predictions: {is_close}")
