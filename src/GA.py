import numpy as np
import pandas as pd
import streamlit as st
import random
import os
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def calculate_mse(actual, predicted):
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    mse = np.mean(squared_errors)  # Calculate mean of squared errors
    return mse

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

def evaluate_network(chromosome, activation_function):
    w1, w2, w3, b = chromosome
    total_error = 0
    predictions = []
    for i in range(len(inputs)):
        hidden_layer_input = np.dot(inputs[i], np.array([[w1, w2], [w2, w3]])) + b
        hidden_layer_output = activation_function(hidden_layer_input)
        final_output = sigmoid(np.dot(hidden_layer_output, np.ones((2, 1))))
        predictions.append(final_output[0])
        total_error += (outputs[i] - final_output) ** 2
    mse = calculate_mse(outputs.flatten(), predictions)
    return mse, predictions

population_size = 100
generations = 1000
mutation_rate = 0.01

def initialize_population(size):
    return [np.random.rand(4) for _ in range(size)]

def select(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        i, j = np.random.choice(len(population), 2)
        if fitnesses[i] < fitnesses[j]:
            selected.append(population[i])
        else:
            selected.append(population[j])
    return selected

def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        return np.concatenate((parent1[:point], parent2[point:]))
    return parent1

def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] += np.random.normal()
    return chromosome

def genetic_algorithm(crossover_rate, activation_function):
    population = initialize_population(population_size)
    best_chromosome = None
    best_fitness = float('inf')
    results = []

    for generation in range(generations):
        fitnesses = [evaluate_network(chromosome, activation_function)[0] for chromosome in population]

        for i in range(len(population)):
            if fitnesses[i] < best_fitness:
                best_fitness = fitnesses[i]
                best_chromosome = population[i]

        for chromosome, fitness in zip(population, fitnesses):
            _, predictions = evaluate_network(chromosome, activation_function)
            runtime = time.time() - start_time
            results.append((generation, fitness, chromosome, predictions, runtime))

        population = select(population, fitnesses)

        new_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % len(population)]
            child1 = crossover(parent1, parent2, crossover_rate)
            child2 = crossover(parent2, parent1, crossover_rate)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population

    return best_chromosome, best_fitness, results

st.title("N.N Demonstration to solve XOR")
st.write(" ")
st.write("In this section, you can use the **Genetic Algorithm**, in order to solve the **XOR problem**, while training a **Neural Network!**")
st.write("You can adjust the hyperparameters of the algorithm for your best fit, using the **toolbox** below:")


with st.expander("Adjust Hyperparameters", expanded=True):
    population_size = st.slider("Population Size", min_value=10, max_value=200, value=100, step=10)
    generations = st.slider("Number of Generations", min_value=100, max_value=5000, value=1000, step=100)
    mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
    
    use_multiple_crossover = st.checkbox("Use Multiple Crossover Rates", value=False)
    if use_multiple_crossover:
        crossover_range = st.slider("Crossover Range", min_value=0.0, max_value=1.0, value=(0.1, 0.7), step=0.1)
        crossover_step = st.slider("Crossover Step", min_value=0.01, max_value=0.1, value=0.1, step=0.01)
    else:
        crossover_rate = st.slider("Crossover Rate", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    activation_function_name = st.selectbox("Choose Activation Function", ["sigmoid", "relu", "tanh"], key="selection")
    if activation_function_name == "sigmoid":
        activation_function = sigmoid
    elif activation_function_name == "relu":
        activation_function = relu
    else:
        activation_function = tanh

if st.button("Optimize Neural Network", key="button"):
    with st.spinner('Computing...'):
        start_time = time.time()
        if use_multiple_crossover:
            all_results = []
            for rate in np.arange(crossover_range[0], crossover_range[1] + crossover_step, crossover_step):
                best_chromosome, best_fitness, results = genetic_algorithm(rate, activation_function)
                all_results.extend(results)
        else:
            best_chromosome, best_fitness, results = genetic_algorithm(crossover_rate, activation_function)
            all_results = results
        end_time = time.time()

        best_result = min(all_results, key=lambda x: x[1])
        best_generation, best_loss, best_weights, best_output, best_runtime = best_result

        results_df = pd.DataFrame(all_results, columns=["Generation", "Loss", "Best Weights", "Predicted Values", "Runtime"])
        st.write("### Results for Different Generations:")
        st.write(results_df.drop(columns=["Best Weights"]))

        st.write("### Actual vs Predicted Results (Best Configuration):")
        result_table = np.hstack((inputs, outputs, np.array(best_output).reshape(-1, 1)))
        result_df = pd.DataFrame(result_table, columns=["Input 1", "Input 2", "Actual Output", "Predicted Output"])
        st.write(result_df)

        results_dir = "/mount/src/metaml/src/results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        csv_best_result_df = pd.DataFrame({
            "Input 1": inputs[:, 0],
            "Input 2": inputs[:, 1],
            "Output": best_output,
            "Expected Output": outputs.flatten(),
            "Generation": [best_generation] * len(inputs),
            "Runtime": [best_runtime] * len(inputs)
        })
        csv_best_result_df.to_csv(os.path.join(results_dir, "best_result_ga.csv"), index=False)

        csv_all_results_df = pd.DataFrame({
            "Generation": results_df["Generation"],
            "Loss": results_df["Loss"],
            "Runtime": results_df["Runtime"]
        })
        csv_all_results_df.to_csv(os.path.join(results_dir, "all_results_ga.csv"), index=False)

        config = {
            "config": {
                "generation": best_generation,
                "loss": best_loss,
                "mutation_rate": mutation_rate,
                "crossover_rate": crossover_rate if not use_multiple_crossover else list(np.arange(crossover_range[0], crossover_range[1] + crossover_step, crossover_step)),
                "population_size": population_size,
                "number_of_generations": generations,
                "runtime": best_runtime
            }
        }
        st.expander("Best Configuration", expanded=True).write(config)
