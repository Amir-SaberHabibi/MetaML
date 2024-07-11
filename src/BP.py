import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Function to calculate Mean Squared Error
def calculate_mse(actual, predicted):
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    mse = np.mean(squared_errors)
    return mse

# XOR inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

st.title("Backpropagation - XOR")

# Hyperparameter input in the sidebar
learning_rate_range = st.sidebar.slider("Learning Rate Range", min_value=0.01, max_value=1.0, value=(0.01, 0.1), step=0.01)
learning_rate_step = st.sidebar.slider("Learning Rate Step", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
epochs = st.sidebar.slider("Number of Epochs", min_value=1000, max_value=20000, value=10000, step=1000)

if st.sidebar.button("Compute"):
    with st.spinner('Computing...'):
        all_results = []

        best_mse = float('inf')
        best_mse_df = None  # --> dataframe to store in the results table
        best_predictions = None
        best_lr = None

        for lr in np.arange(learning_rate_range[0], learning_rate_range[1] + learning_rate_step, learning_rate_step):
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
                    hidden_layer_output = sigmoid(hidden_layer_input)
                    final_output_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
                    final_output = sigmoid(final_output_input)

                    # Calculate Error
                    error = outputs[i] - final_output

                    # Backward Propagation
                    d_final_output = error * sigmoid_derivative(final_output)
                    error_hidden_layer = d_final_output.dot(weights_hidden_output.T)
                    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

                    # Update Weights and Biases
                    weights_hidden_output += hidden_layer_output.T.dot(d_final_output) * lr
                    weights_input_hidden += inputs[i].reshape(-1, 1).dot(d_hidden_layer) * lr
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
                hidden_layer_output = sigmoid(hidden_layer_input)
                final_output_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
                final_output = sigmoid(final_output_input)
                final_outputs.append(final_output[0, 0])  # Extract the scalar value

            for i in range(len(inputs)):
                all_results.append({
                    "Learning Rate": lr,
                    "Input 1": inputs[i][0],
                    "Input 2": inputs[i][1],
                    "Predicted Output": final_outputs[i],
                    "Expected Output": outputs[i][0]
                })

            # Save best results
            last_mse = mse_data[-1][1]
            if last_mse < best_mse:
                best_mse = last_mse
                best_mse_df = pd.DataFrame(mse_data, columns=["Epoch", "Mean Squared Error"])
                best_predictions = final_outputs
                best_lr = lr

        # Convert results to DataFrame
        all_results_df = pd.DataFrame(all_results)

        # Display all results
        st.subheader("All Results for Different Learning Rates")
        st.dataframe(all_results_df)

        # Display best MSE Data
        st.subheader("Best Training Progress (Mean Squared Error)")
        st.dataframe(best_mse_df)

        # Plot the best Mean Squared Error over epochs
        fig = px.line(best_mse_df, x="Epoch", y="Mean Squared Error", title="Best Training Progress (Mean Squared Error)")
        st.plotly_chart(fig)

        # Display best prediction results
        st.subheader("Best Prediction Results Compared with Actual Results")
        comparison_df = pd.DataFrame({
            "Input 1": inputs[:, 0],
            "Input 2": inputs[:, 1],
            "Predicted Output": best_predictions,
            "Expected Output": outputs.flatten()
        })
        st.dataframe(comparison_df)

        # Save best results to CSV
        csv_result_df = pd.DataFrame({
            "Input 1": inputs[:, 0],
            "Input 2": inputs[:, 1],
            "Output": best_predictions,
            "Expected Output": outputs.flatten()
        })
        csv_result_df.to_csv("best_result_bp.csv", index=False)

        # Display best configuration in JSON format
        route = {
            "config": {
                "learning_rate_range": learning_rate_range,
                "learning_rate_step": learning_rate_step,
                "epochs": epochs,
                "best_learning_rate": best_lr,
            }
        }
        st.expander("Export shortest path route", expanded=True).write(route)
