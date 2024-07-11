import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(df):
    mse = ((df['Output'] - df['Expected Output']) ** 2).mean()
    return mse

# Main function
def main():
    st.title("Comparison of BP and PSO Performance on XOR Problem")

    # Read the CSV files
    bp_results = pd.read_csv("best_result_bp.csv")
    pso_results = pd.read_csv("best_result_pso.csv")

    # Calculate MSE for BP and PSO
    mse_bp = calculate_mse(bp_results)
    mse_pso = calculate_mse(pso_results)

    # Display MSE values
    st.write(f"Mean Squared Error (BP): `{mse_bp}`")
    st.write(f"Mean Squared Error (PSO): `{mse_pso}`")

    # Compare performance on a linear graph
    algorithms = ['BP', 'PSO']
    mse_values = [mse_bp, mse_pso]

    fig = go.Figure(data=go.Scatter(x=algorithms, y=mse_values, mode='lines+markers'))
    fig.update_layout(
        title="Performance Comparison of BP and PSO",
        xaxis_title="Algorithm",
        yaxis_title="Mean Squared Error (MSE)"
    )

    st.plotly_chart(fig)

# Run the main function
if __name__ == "__main__":
    main()
