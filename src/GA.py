import numpy as np
import pandas as pd
import streamlit as st
import random

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to calculate Mean Squared Error
def calculate_mse(actual, predicted):
    squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
    mse = np.mean(squared_errors)  # Calculate mean of squared errors
    return mse

# XOR inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Evaluate the neural network with given weights and biases
def evaluate_network(chromosome):
    w1, w2, w3, b = chromosome
    total_error = 0
    predictions = []
    for i in range(len(inputs)):
        hidden_layer_input = np.dot(inputs[i], np.array([[w1, w2], [w2, w3]])) + b
        hidden_layer_output = sigmoid(hidden_layer_input)
        final_output = sigmoid(np.dot(hidden_layer_output, np.ones((2, 1))))
        predictions.append(final_output[0])
        total_error += (outputs[i] - final_output) ** 2
    mse = calculate_mse(outputs.flatten(), predictions)
    return mse, predictions

# Genetic Algorithm Parameters
population_size = 100
generations = 1000
mutation_rate = 0.01
crossover_rate = 0.7

# Initialize the population
def initialize_population(size):
    return [np.random.rand(4) for _ in range(size)]

# Selection: Tournament Selection
def select(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        i, j = np.random.choice(len(population), 2)
        if fitnesses[i] < fitnesses[j]:
            selected.append(population[i])
        else:
            selected.append(population[j])
    return selected

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        return np.concatenate((parent1[:point], parent2[point:]))
    return parent1

# Mutation: Randomly mutate genes
def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] += np.random.normal()
    return chromosome

# Main Genetic Algorithm loop
def genetic_algorithm():
    population = initialize_population(population_size)
    best_chromosome = None
    best_fitness = float('inf')

    for generation in range(generations):
        fitnesses = [evaluate_network(chromosome)[0] for chromosome in population]

        # Find the best solution in the current population
        for i in range(len(population)):
            if fitnesses[i] < best_fitness:
                best_fitness = fitnesses[i]
                best_chromosome = population[i]

        # Selection
        population = select(population, fitnesses)

        # Crossover and Mutation
        new_population = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % len(population)]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population

    return best_chromosome, best_fitness

st.title("Genetic Algorithm Neural Network XOR Solution")

# GA Parameters input
population_size = st.slider("Population Size", min_value=10, max_value=200, value=100, step=10)
generations = st.slider("Number of Generations", min_value=100, max_value=5000, value=1000, step=100)
mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
crossover_rate = st.slider("Crossover Rate", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

if st.button("Compute"):
    best_chromosome, best_fitness = genetic_algorithm()
    mse, ga_predictions = evaluate_network(best_chromosome)

    st.write(f"Best Chromosome: {best_chromosome}")
    st.write(f"Best Fitness: {best_fitness}")
    st.write(f"Mean Squared Error: {mse}")

    # Display final GA outputs
    ga_outputs_df = pd.DataFrame(np.hstack((inputs, np.array(ga_predictions).reshape(-1, 1))), columns=["Input 1", "Input 2", "Output"])
    ga_outputs_df['Expected Output'] = outputs.flatten()
    ga_outputs_df.to_csv("ga_results.csv", index=False)  # Save the results for comparison

    st.subheader("Final Outputs for XOR Inputs (GA)")
    st.dataframe(ga_outputs_df)
