import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import time
import os

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Function to calculate Mean Squared Error
def calculate_mse(actual, predicted):
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    mse = np.mean(squared_errors)
    return mse

# XOR inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

st.title("Backpropagation - XOR")
st.write(" ")

# Hyperparameter input using expanders
with st.expander("Adjust Hyperparameters", expanded=True):
    use_single_lr = st.checkbox("Use a single learning rate instead of a range")
    
    if use_single_lr:
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    else:
        learning_rate_range = st.slider("Learning Rate Range", min_value=0.01, max_value=1.0, value=(0.01, 0.1), step=0.01)
        learning_rate_step = st.slider("Learning Rate Step", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
        
    epochs = st.slider("Number of Epochs", min_value=1000, max_value=20000, value=10000, step=1000)
    activation_function = st.selectbox("Activation Function", ("Sigmoid", "Tanh", "ReLU"))

# Select activation function
if activation_function == "Sigmoid":
    activation_func = sigmoid
    activation_derivative_func = sigmoid_derivative
elif activation_function == "Tanh":
    activation_func = tanh
    activation_derivative_func = tanh_derivative
else:
    activation_func = relu
    activation_derivative_func = relu_derivative

if st.button("Compute"):
    with st.spinner('Computing...'):
        start_time = time.time()
        results = []

        if use_single_lr:
            learning_rates = [learning_rate]
        else:
            learning_rates = np.arange(learning_rate_range[0], learning_rate_range[1] + learning_rate_step, learning_rate_step)

        for lr in learning_rates:
            # Initialize weights and biases
            weights_input_hidden = np.random.rand(2, 2)
            weights_hidden_output = np.random.rand(2, 1)
            bias_hidden = np.random.rand(1, 2)
            bias_output = np.random.rand(1, 1)

            mse_data = []

            for epoch in range(epochs):
                epoch_predictions = []
                for i in range(len(inputs)):
                    # Forward Propagation
                    hidden_layer_input = np.dot(inputs[i], weights_input_hidden) + bias_hidden
                    hidden_layer_output = activation_func(hidden_layer_input)
                    final_output_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
                    final_output = activation_func(final_output_input)

                    # Calculate Error
                    error = outputs[i] - final_output

                    # Backward Propagation
                    d_final_output = error * activation_derivative_func(final_output)
                    error_hidden_layer = d_final_output.dot(weights_hidden_output.T)
                    d_hidden_layer = error_hidden_layer * activation_derivative_func(hidden_layer_output)

                    # Update Weights and Biases
                    weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * lr
                    weights_input_hidden += np.outer(inputs[i], d_hidden_layer) * lr
                    bias_output += np.sum(d_final_output, axis=0, keepdims=True) * lr
                    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

                    epoch_predictions.append(final_output[0, 0])

                # Calculate mean squared error for the epoch using the provided function
                mse = calculate_mse(outputs, epoch_predictions)
                mse_data.append([epoch, mse])

            # Store results
            final_outputs = []
            for i in range(len(inputs)):
                hidden_layer_input = np.dot(inputs[i], weights_input_hidden) + bias_hidden
                hidden_layer_output = activation_func(hidden_layer_input)
                final_output_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
                final_output = activation_func(final_output_input)
                final_outputs.append(final_output[0, 0])  # Extract the scalar value

            runtime = time.time() - start_time

            # Record results
            results.append((lr, mse_data[-1][1], weights_input_hidden, final_outputs, runtime))

        # Find best result
        best_result = min(results, key=lambda x: x[1])
        best_lr, best_mse, best_weights, best_output, best_runtime = best_result

        # Display results table
        results_df = pd.DataFrame(results, columns=["Learning Rate", "MSE", "Best Weights", "Predicted Values", "Runtime"])
        st.write("### Results for Different Learning Rates:")
        st.dataframe(results_df.drop(columns=["Best Weights"]))

        # Display table of actual vs predicted results for the best configuration
        st.write("### Actual vs Predicted Results (Best Configuration):")
        result_table = np.hstack((inputs, outputs, np.array(best_output).reshape(-1, 1)))
        result_df = pd.DataFrame(result_table, columns=["Input 1", "Input 2", "Actual Output", "Predicted Output"])
        st.dataframe(result_df)


        # Save results to CSV
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Save best configuration results
        csv_best_result_df = pd.DataFrame({
            "Input 1": inputs[:, 0],
            "Input 2": inputs[:, 1],
            "Output": np.array(best_output).flatten(),
            "Expected Output": outputs.flatten(),
            "Runtime": [best_runtime] * len(inputs)
        })
        csv_best_result_df.to_csv(os.path.join(results_dir, "best_result_bp.csv"), index=False)

        # Save all configurations results
        csv_all_results_df = pd.DataFrame({
            "Learning Rate": results_df["Learning Rate"],
            "MSE": results_df["MSE"],
            "Runtime": results_df["Runtime"]
        })
        csv_all_results_df.to_csv(os.path.join(results_dir, "all_results_bp.csv"), index=False)

        # Display best configuration in JSON format
        route = {
            "config": {
                "learning_rate": best_lr,
                "error": best_mse,
                "epochs": epochs,
                "activation_function": activation_function,
                "best_weights": best_weights.tolist(),
                "runtime": best_runtime
            }
        }
        st.expander("Best Configuration", expanded=True).write(route)
