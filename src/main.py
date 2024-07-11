import streamlit as st

st.title("Neural Network XOR Solution")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Backpropagation", "Genetic Algorithm", "PSO", "Comparison"])

if page == "Backpropagation":
    st.header("Backpropagation Algorithm")

    st.write("""
    **Features of the Backpropagation Algorithm:**

    1. **Initialization**:
       - Weights and biases are initialized randomly.
       - The learning rate and the number of epochs are set as hyperparameters.

    2. **Forward Propagation**:
       - Inputs are passed through the network to obtain outputs using the sigmoid activation function.
       - The output of each layer is calculated and used as input for the next layer.

    3. **Error Calculation**:
       - The error is calculated as the difference between the predicted outputs and the actual XOR outputs.
       - The Mean Squared Error (MSE) is used as the error metric.

    4. **Backward Propagation**:
       - The error is propagated back through the network.
       - The weights and biases are adjusted to minimize the error using the gradient of the error.

    5. **Iteration**:
       - The process of forward propagation, error calculation, and backward propagation is repeated for a specified number of epochs.

    6. **Result**:
       - The algorithm returns the final weights and biases, and the corresponding error value.
    """)
    exec(open("BP.py").read())
elif page == "Genetic Algorithm":
    exec(open("GA.py").read())
elif page == "Comparison":
    exec(open("comparison.py").read())
elif page == "PSO":
    st.header("Particle Swarm Optimization (PSO) Algorithm")
    
    st.write("""
    **Features of the PSO Algorithm:**
    
    1. **Initialization**: 
       - A population of particles is initialized, each representing a potential solution (weights and biases for the neural network).
       - Each particle has a position in the search space and a velocity.

    2. **Evaluation**:
       - Each particle's position (solution) is evaluated using the Mean Squared Error (MSE) between the predicted outputs and the actual XOR outputs.

    3. **Personal and Global Bests**:
       - The algorithm keeps track of each particle's personal best position (the position that yielded the lowest error for that particle) and the global best position (the position that yielded the lowest error among all particles).

    4. **Update Velocities and Positions**:
       - In each iteration, the velocity of each particle is updated based on three components:
         - `Inertia`: The previous velocity, scaled by the inertia weight.
         - `Cognitive` Component: Steers the particle towards its personal best position.
         - `Social Component`: Steers the particle towards the global best position.
       - The particle's position is then updated by adding the new velocity to the current position.

    5. **Iteration**:
       - The process of evaluating, updating velocities, and updating positions is repeated for a specified number of generations.
       - The algorithm converges when the global best position no longer changes significantly or when the maximum number of generations is reached.

    6. **Result**:
       - The algorithm returns the global best position (the set of weights and biases that minimize the error) and the corresponding error value.
    """)

    exec(open("PSO.py").read())
