# PyTorch-Linear-DNN-Fundamentals

This project serves as a step-by-step introduction to the PyTorch deep learning workflow, demonstrating how a simple neural network learns the pattern of a straight line (`y = weight * x + bias`).

It implements the full cycle of data preparation, model construction, training, evaluation, and saving/loading, following a device-agnostic approach (works on both CPU and GPU).

## Core Concepts Explained

A deep learning workflow typically involves these steps:

1.  **Data Preparation:** Create and split the dataset (training, testing).
2.  **Model Building:** Define the neural network architecture.
3.  **Loss Function & Optimizer:** Define how the model measures errors (Loss) and how it updates its parameters to reduce those errors (Optimizer).
4.  **Training Loop:** The iterative process of _Forward Pass_, _Loss Calculation_, _Backpropagation_, and _Optimizer Step_.
5.  **Inference:** Using the trained model to make predictions.
6.  **Saving/Loading:** Persisting the learned parameters.

## Files

-   `linear_regression_pytorch.py`: The main, self-contained Python script implementing the entire workflow.
-   `requirements.txt`: Lists necessary Python libraries.

## Setup and Running

### 1. Prerequisites

You need Python (3.8+) and the following libraries:

```
torch
matplotlib
```

### 2. Run Locally (Recommended)

1.  Clone the repository:
    
    ```
    git clone [https://github.com/your-username/PyTorch-Linear-DNN-Fundamentals.git](https://github.com/your-username/PyTorch-Linear-DNN-Fundamentals.git)
    cd PyTorch-Linear-DNN-Fundamentals
    ```
    
2.  Install dependencies:
    
    ```
    pip install -r requirements.txt
    ```
    
3.  Run the script:
    
    ```
    python linear_regression_pytorch.py
    ```
    
    The script will print training progress, model parameters, and save plots/the final model state.