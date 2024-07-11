import streamlit as st
import numpy as np
import pandas as pd

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))

        # Initialize biases
        self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, self.output_size))

    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def compute_loss(self, y_true):
        return np.mean((y_true - self.output) ** 2)

# PSO parameters
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
        for iteration in range(self.n_iterations):
            for particle in self.particles:
                # Update neural network weights and biases
                weights, biases = np.split(particle.position, [self.neural_network.input_size * self.neural_network.hidden_size + self.neural_network.hidden_size * self.neural_network.output_size])
                weight_split = np.split(weights, [self.neural_network.input_size * self.neural_network.hidden_size])
                self.neural_network.weights_input_hidden = weight_split[0].reshape((self.neural_network.input_size, self.neural_network.hidden_size))
                self.neural_network.weights_hidden_output = weight_split[1].reshape((self.neural_network.hidden_size, self.neural_network.output_size))
                bias_split = np.split(biases, [self.neural_network.hidden_size])
                self.neural_network.bias_hidden = bias_split[0].reshape((1, self.neural_network.hidden_size))
                self.neural_network.bias_output = bias_split[1].reshape((1, self.neural_network.output_size))
                
                # Forward pass
                output = self.neural_network.forward(X)
                
                # Compute loss
                loss = self.neural_network.compute_loss(y)
                
                # Update particle's best position
                if loss < particle.best_score:
                    particle.best_score = loss
                    particle.best_position = particle.position.copy()
                    
                # Update global best position
                if loss < self.global_best_score:
                    self.global_best_score = loss
                    self.global_best_position = particle.position.copy()
                    
            for particle in self.particles:
                # Update velocity
                r1, r2 = np.random.rand(2)
                
                particle.velocity = (self.inertia * particle.velocity + 
                                     self.cognitive_component * r1 * (particle.best_position - particle.position) +
                                     self.social_component * r2 * (self.global_best_position - particle.position))
                
                # Update position
                particle.position += particle.velocity
                
        return self.global_best_position, self.global_best_score

# Streamlit app
st.title("Neural Network Training with PSO")

# Initialize neural network and PSO
st.sidebar.header("PSO Parameters")
n_particles = st.sidebar.slider("Number of Particles", 10, 100, 30)
n_iterations = st.sidebar.slider("Number of Iterations", 100, 5000, 1000)
hidden_size = st.sidebar.slider("Hidden Layer Size", 1, 10, 5)
input_size = 2
output_size = 1
inertia_range = st.sidebar.slider("Inertia Range", min_value=0.0, max_value=2.0, value=(0.01, 0.1), step=0.02)
inertia_step = st.sidebar.slider("Inertia Step", min_value=0.01, max_value=0.1, value=0.01, step=0.01)
cognitive_component = st.sidebar.slider("Cognitive Component", 0.1, 3.0, 2.05)
social_component = st.sidebar.slider("Social Component", 0.1, 3.0, 2.05)
dim = input_size * hidden_size + hidden_size * output_size + hidden_size + output_size

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

results = []

nn = NeuralNetwork(input_size, hidden_size, output_size)

# Iterate over the inertia values
if st.sidebar.button("Optimize Neural Network"):
    with st.spinner('Computing...'):
        for inertia in np.arange(inertia_range[0], inertia_range[1] + inertia_step, inertia_step):
            pso = PSO(n_particles, dim, n_iterations, nn, inertia, cognitive_component, social_component)
            best_weights, best_score = pso.optimize(X, y)
            
            # Set optimized weights to the neural network
            weights, biases = np.split(best_weights, [input_size * hidden_size + hidden_size * output_size])
            weight_split = np.split(weights, [input_size * hidden_size])
            nn.weights_input_hidden = weight_split[0].reshape((input_size, hidden_size))
            nn.weights_hidden_output = weight_split[1].reshape((hidden_size, output_size))
            bias_split = np.split(biases, [hidden_size])
            nn.bias_hidden = bias_split[0].reshape((1, hidden_size))
            nn.bias_output = bias_split[1].reshape((1, output_size))
            
            # Test the neural network
            output = nn.forward(X)
            
            # Store results including predicted values (flattened)
            results.append((inertia, best_score, best_weights, output.flatten()))

        # Find the best result
        best_result = min(results, key=lambda x: x[1])
        best_inertia, best_loss, best_weights, best_output = best_result
        
        # Display results table
        results_df = pd.DataFrame(results, columns=["Inertia", "Loss", "Best Weights", "Predicted Values"])
        st.write("### Results for Different Inertia Values")
        st.write(results_df.drop(columns=["Best Weights"]))


        # Display table of actual vs predicted results for the best configuration
        st.write("### Actual vs Predicted Results (Best Configuration)")
        result_table = np.hstack((X, y, best_output.reshape(-1, 1)))
        result_df = pd.DataFrame(result_table, columns=["Input 1", "Input 2", "Actual Output", "Predicted Output"])
        st.write(result_df)
        
        # Save results to CSV
        csv_result_df = pd.DataFrame({
            "Input 1": X[:, 0],
            "Input 2": X[:, 1],
            "Output": best_output,
            "Expected Output": y.flatten()
        })
        csv_result_df.to_csv("best_result_pso.csv", index=False)
        # st.write("### CSV File Created")
        # st.write(csv_result_df)
        
        # Display best configuration in JSON format
        # st.write("### Best Configuration")
        route = {
            "config": {
                "inertia": best_inertia,
                "loss": best_loss,
                "cognitive_component": cognitive_component,
                "social_component": social_component,
                "hidden_layer_size": hidden_size,
                "input_layer_size": input_size,
                "output_layer_size": output_size,
                "number_of_iterations": n_iterations,
                "number_of_particles": n_particles            }
        }
        st.expander("Best Configuration", expanded=True).write(route)
