import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os

def calculate_mse(df):
    mse = ((df['Output'] - df['Expected Output']) ** 2).mean()
    return mse

def calculate_avg_runtime(df):
    avg_runtime = df['Runtime'].mean()
    return avg_runtime

def update_csv_with_mse(csv_path, mse_value):
    df = pd.read_csv(csv_path)
    df['MSE'] = mse_value
    df.to_csv(csv_path, index=False)

def main():
    st.title("Comparison of BP, PSO, and GA Performance on XOR Problem")
    st.write(" ")
    st.write("In this section, you can observe how the algorithms perform after processing. This helps you find the best solution possible for your defined problem!")

    bp_path = "/mount/src/metaml/src/results/best_result_bp.csv"
    pso_path = "/mount/src/metaml/src/results/best_result_pso.csv"
    ga_path = "/mount/src/metaml/src/results/best_result_ga.csv"

    try:
        bp_results = pd.read_csv(bp_path)
        pso_results = pd.read_csv(pso_path)
        ga_results = pd.read_csv(ga_path)

        mse_bp = calculate_mse(bp_results)
        mse_pso = calculate_mse(pso_results)
        mse_ga = calculate_mse(ga_results)

        update_csv_with_mse(bp_path, mse_bp)
        update_csv_with_mse(pso_path, mse_pso)
        update_csv_with_mse(ga_path, mse_ga)

        avg_runtime_bp = calculate_avg_runtime(bp_results)
        avg_runtime_pso = calculate_avg_runtime(pso_results)
        avg_runtime_ga = calculate_avg_runtime(ga_results)

        data = {
            'Algorithm': ['BP', 'PSO', 'GA'],
            'Mean Squared Error': [mse_bp, mse_pso, mse_ga],
            'Average Runtime (seconds)': [avg_runtime_bp, avg_runtime_pso, avg_runtime_ga]
        }
        df = pd.DataFrame(data)

        st.write(df)

        algorithms = ['BP', 'PSO', 'GA']
        mse_values = [mse_bp, mse_pso, mse_ga]
        avg_runtime_values = [avg_runtime_bp, avg_runtime_pso, avg_runtime_ga]

        fig_mse = go.Figure(data=go.Scatter(x=algorithms, y=mse_values, mode='lines+markers'))
        fig_mse.update_layout(
            title="MSE Performance Comparison: BP vs PSO vs GA",
            xaxis_title="Algorithm",
            yaxis_title="Mean Squared Error (MSE)"
        )

        st.plotly_chart(fig_mse)

        fig_runtime = go.Figure(data=go.Scatter(x=algorithms, y=avg_runtime_values, mode='lines+markers'))
        fig_runtime.update_layout(
            title="Runtime Performance Comparison: BP vs PSO vs GA",
            xaxis_title="Algorithm",
            yaxis_title="Average Runtime (seconds)"
        )

        st.plotly_chart(fig_runtime)

    except FileNotFoundError:
        st.error("Please run the algorithm components first to generate the results, and then come back.")
    except Exception as e:
        st.error("An unexpected error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    main()
