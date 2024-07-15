import streamlit as st
import numpy as np
import pandas as pd
import time
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_function='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))

        # Initialize biases
        self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, self.output_size))

        # Set activation function
        self.activation_function = activation_function
        if activation_function == 'sigmoid':
            self.activate = sigmoid
            self.activate_derivative = sigmoid_derivative
        elif activation_function == 'relu':
            self.activate = relu
            self.activate_derivative = relu_derivative
        elif activation_function == 'tanh':
            self.activate = tanh
            self.activate_derivative = tanh_derivative

    def forward(self, X):
        self.hidden = self.activate(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.activate(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def compute_loss(self, y_true):
        return np.mean((y_true - self.output) ** 2)

class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(-1, 1, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

class PSO:
    def __init__(self, n_particles, dim, n_iterations, neural_network, inertia, cognitive_component, social_component):
        self.n_particles = n_particles
        self.dim = dim
        self.n_iterations = n_iterations
        self.neural_network = neural_network
        self.inertia = inertia
        self.cognitive_component = cognitive_component
        self.social_component = social_component
        self.particles = [Particle(dim) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_score = float('inf')

    def optimize(self, X, y):
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            for particle in self.particles:
                weights, biases = np.split(particle.position, [self.neural_network.input_size * self.neural_network.hidden_size + self.neural_network.hidden_size * self.neural_network.output_size])
                weight_split = np.split(weights, [self.neural_network.input_size * self.neural_network.hidden_size])
                self.neural_network.weights_input_hidden = weight_split[0].reshape((self.neural_network.input_size, self.neural_network.hidden_size))
                self.neural_network.weights_hidden_output = weight_split[1].reshape((self.neural_network.hidden_size, self.neural_network.output_size))
                bias_split = np.split(biases, [self.neural_network.hidden_size])
                self.neural_network.bias_hidden = bias_split[0].reshape((1, self.neural_network.hidden_size))
                self.neural_network.bias_output = bias_split[1].reshape((1, self.neural_network.output_size))
                
                output = self.neural_network.forward(X)
                
                loss = self.neural_network.compute_loss(y)
                
                if loss < particle.best_score:
                    particle.best_score = loss
                    particle.best_position = particle.position.copy()
                    
                if loss < self.global_best_score:
                    self.global_best_score = loss
                    self.global_best_position = particle.position.copy()
                    
            for particle in self.particles:
                r1, r2 = np.random.rand(2)
                
                particle.velocity = (self.inertia * particle.velocity + 
                                     self.cognitive_component * r1 * (particle.best_position - particle.position) +
                                     self.social_component * r2 * (self.global_best_position - particle.position))
                
                particle.position += particle.velocity
        
        end_time = time.time()
        runtime = end_time - start_time
        return self.global_best_position, self.global_best_score, runtime

st.title("N.N Demonstration to solve XOR")
st.write(" ")
st.write("In this section, you can use the **Particle Swarm Optimization Algorithm**, in order to solve the **XOR problem**, while training a **Neural Network!**")
st.write("You can adjust the hyperparameters of the algorithm for your best fit, using the **toolbox** below:")

with st.expander("Adjust Hyperparameters", expanded=True):
    n_particles = st.slider("Number of Particles", 10, 100, 30)
    n_iterations = st.slider("Number of Iterations", 100, 5000, 1000)
    hidden_size = st.slider("Hidden Layer Size", 1, 10, 5)
    input_size = 2
    output_size = 1

    use_multiple_inertia = st.checkbox("Use Multiple Inertia Values")

    if use_multiple_inertia:
        inertia_range = st.slider("Inertia Range", min_value=0.0, max_value=2.0, value=(0.01, 0.1), step=0.02)
        inertia_step = st.slider("Inertia Step", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
    else:
        inertia = st.slider("Inertia", min_value=0.0, max_value=2.0, value=0.1, step=0.01)

    cognitive_component = st.slider("Cognitive Component", 0.1, 3.0, 2.05)
    social_component = st.slider("Social Component", 0.1, 3.0, 2.05)
    activation_function = st.selectbox("Choose Activation Function", ["sigmoid", "relu", "tanh"])
    dim = input_size * hidden_size + hidden_size * output_size + hidden_size + output_size

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

results = []

nn = NeuralNetwork(input_size, hidden_size, output_size, activation_function)

if st.button("Optimize Neural Network"):
    with st.spinner('Computing...'):
        if use_multiple_inertia:
            for inertia in np.arange(inertia_range[0], inertia_range[1] + inertia_step, inertia_step):
                pso = PSO(n_particles, dim, n_iterations, nn, inertia, cognitive_component, social_component)
                best_weights, best_score, runtime = pso.optimize(X, y)
                
                weights, biases = np.split(best_weights, [input_size * hidden_size + hidden_size * output_size])
                weight_split = np.split(weights, [input_size * hidden_size])
                nn.weights_input_hidden = weight_split[0].reshape((input_size, hidden_size))
                nn.weights_hidden_output = weight_split[1].reshape((hidden_size, output_size))
                bias_split = np.split(biases, [hidden_size])
                nn.bias_hidden = bias_split[0].reshape((1, hidden_size))
                nn.bias_output = bias_split[1].reshape((1, output_size))
                
                output = nn.forward(X)
                
                results.append((inertia, best_score, best_weights, output.flatten(), runtime))
        else:
            pso = PSO(n_particles, dim, n_iterations, nn, inertia, cognitive_component, social_component)
            best_weights, best_score, runtime = pso.optimize(X, y)
            
            weights, biases = np.split(best_weights, [input_size * hidden_size + hidden_size * output_size])
            weight_split = np.split(weights, [input_size * hidden_size])
            nn.weights_input_hidden = weight_split[0].reshape((input_size, hidden_size))
            nn.weights_hidden_output = weight_split[1].reshape((hidden_size, output_size))
            bias_split = np.split(biases, [hidden_size])
            nn.bias_hidden = bias_split[0].reshape((1, hidden_size))
            nn.bias_output = bias_split[1].reshape((1, output_size))
            
            output = nn.forward(X)
            
            results.append((inertia, best_score, best_weights, output.flatten(), runtime))

        best_result = min(results, key=lambda x: x[1])
        best_inertia, best_loss, best_weights, best_output, best_runtime = best_result
        
        results_df = pd.DataFrame(results, columns=["Inertia", "Loss", "Best Weights", "Predicted Values", "Runtime"])
        st.write("### Results for Different Inertia Values:")
        st.write(results_df.drop(columns=["Best Weights"]))

        st.write("### Actual vs Predicted Results (Best Configuration):")
        result_table = np.hstack((X, y, best_output.reshape(-1, 1)))
        result_df = pd.DataFrame(result_table, columns=["Input 1", "Input 2", "Actual Output", "Predicted Output"])
        st.write(result_df)
        
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        csv_best_result_df = pd.DataFrame({
            "Input 1": X[:, 0],
            "Input 2": X[:, 1],
            "Output": best_output.flatten(),
            "Expected Output": y.flatten(),
            "Runtime": [best_runtime] * len(X)
        })
        csv_best_result_df.to_csv(os.path.join(results_dir, "best_result_pso.csv"), index=False)
        
        csv_all_results_df = pd.DataFrame({
            "Inertia": results_df["Inertia"],
            "Loss": results_df["Loss"],
            "Runtime": results_df["Runtime"]
        })
        csv_all_results_df.to_csv(os.path.join(results_dir, "all_results_pso.csv"), index=False)
        
        config = {
            "config": {
                "inertia": best_inertia,
                "loss": best_loss,
                "cognitive_component": cognitive_component,
                "social_component": social_component,
                "hidden_layer_size": hidden_size,
                "input_layer_size": input_size,
                "output_layer_size": output_size,
                "number_of_iterations": n_iterations,
                "number_of_particles": n_particles,
                "activation_function": activation_function,
                "runtime": best_runtime
            }
        }
        st.expander("Best Configuration", expanded=True).write(config)
