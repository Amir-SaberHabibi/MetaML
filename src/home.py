import streamlit as st


st.set_page_config(page_title="MetaML",
                   page_icon=":car:",
                   layout="centered")

with open("/mount/src/metaml/src//mount/src/metaml/src/ui/sidebar.md", "r") as sidebar_file:
    sidebar_content = sidebar_file.read()

with open("/mount/src/metaml/src//mount/src/metaml/src/ui/styles.md", "r") as styles_file:
    styles_content = styles_file.read()

st.sidebar.markdown(sidebar_content)
st.write(styles_content, unsafe_allow_html=True)

gradient_text_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;700&display=swap');

.gradient-text {
    font-family: 'Lexend', sans-serif;
    font-weight: bold;
    background: -webkit-linear-gradient(left, white, cyan, magenta, yellow);
    background: linear-gradient(to right, white, cyan, magenta, yellow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 10em;
    background-size: 300% 300%;
    animation: float 3s ease-in-out infinite, gradient 5s ease infinite;
}

@keyframes float {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-10px);
    }
}

@keyframes gradient {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}
</style>
<div class="gradient-text">MetaML</div>
"""

st.markdown(gradient_text_html, unsafe_allow_html=True)

# Define tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Home", "Backpropagation", "Genetic Algorithm", "PSO", "Performance Analysis", "Chat", "About Us"])

# Home tab
with tab1:
    st.write(" ")
    st.title('Welcome to MetaheuristicML!')
    st.write(" ")
    st.write("""
    A project focused on optimizing machine learning models using advanced metaheuristic techniques.
    Our algorithms are designed to enhance the efficiency and performance of your machine learning tasks through innovative optimization strategies.
    Explore and compare different optimization methods to find the best fit for your models.
    """)
   #  exec(open("/mount/src/metaml/src/sidebar.py").read())

# Backpropagation tab
with tab2:
    st.write(" ")
    st.header("Backpropagation Algorithm")
    st.write(" ")
    st.markdown("""
      <div style="border: 0.5px solid #4f4e4e; padding: 10px; border-radius: 5px;">
        <h4>Features:</h4>
        <ul>
            <li><b>Initialization</b>: Weights and biases are initialized randomly. The learning rate and the number of epochs are set as hyperparameters.</li>
            <li><b>Forward Propagation</b>: Inputs are passed through the network to obtain outputs using the sigmoid activation function. The output of each layer is calculated and used as input for the next layer.</li>
            <li><b>Error Calculation</b>: The error is calculated as the difference between the predicted outputs and the actual XOR outputs. The Mean Squared Error (MSE) is used as the error metric.</li>
            <li><b>Backward Propagation</b>: The error is propagated back through the network. The weights and biases are adjusted to minimize the error using the gradient of the error.</li>
            <li><b>Iteration</b>: The process of forward propagation, error calculation, and backward propagation is repeated for a specified number of epochs.</li>
            <li><b>Result</b>: The algorithm returns the final weights and biases, and the corresponding error value.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write(" ")
    exec(open("/mount/src/metaml/src/BP.py").read())


# GA tab
with tab3:
    st.write(" ")
    st.header("Genetic Algorithm")
    st.write(" ")
    st.markdown("""
      <div style="border: 0.5px solid #4f4e4e; padding: 10px; border-radius: 5px;">
         <h4>Features:</h4>
         <ul>
            <li><b>Initialization</b>: Random initialization of a population of chromosomes, where each chromosome represents a potential solution (weights and biases for the neural network).</li>
            <li><b>Evaluation</b>: Each chromosome is evaluated using the Mean Squared Error (MSE) between the predicted outputs and the actual XOR outputs.</li>
            <li><b>Selection</b>: Tournament selection is used to select chromosomes from the population based on their fitness (lower MSE is better).</li>
            <li><b>Crossover</b>: Single-point crossover is applied between selected parent chromosomes to create new offspring chromosomes.</li>
            <li><b>Mutation</b>: Random mutation is applied to some genes (parameters) of the chromosomes to introduce new genetic diversity.</li>
            <li><b>Iteration</b>: The process of selection, crossover, and mutation is repeated for a specified number of generations.</li>
            <li><b>Result</b>: The algorithm returns the best chromosome (set of weights and biases that minimize the error) and the corresponding error value.</li>
         </ul>
      </div>
      """, unsafe_allow_html=True)
    st.write(" ")
    exec(open("/mount/src/metaml/src/GA.py").read())


# PSO tab
with tab4:
    st.write(" ")
    st.write("# Particle Swarm Optimization")
    st.write(" ")
    st.markdown("""
      <div style="border: 0.5px solid #4f4e4e; padding: 10px; border-radius: 5px;">
         <h4>Features:</h4>
         <ul>
            <li><b>Initialization</b>: A population of particles is initialized, each representing a potential solution (weights and biases for the neural network). Each particle has a position in the search space and a velocity.</li>
            <li><b>Evaluation</b>: Each particle's position (solution) is evaluated using the Mean Squared Error (MSE) between the predicted outputs and the actual XOR outputs.</li>
            <li><b>Personal and Global Bests</b>: The algorithm keeps track of each particle's personal best position (the position that yielded the lowest error for that particle) and the global best position (the position that yielded the lowest error among all particles).</li>
            <li><b>Update Velocities and Positions</b>: In each iteration, the velocity of each particle is updated based on three components:
                  <ul>
                     <li><code style="color: #5fdae8;">Inertia:</code> The previous velocity, scaled by the inertia weight.</li>
                     <li><code style="color: #5fdae8;">Cognitive Component:</code> Steers the particle towards its personal best position.</li>
                     <li><code style="color: #5fdae8;">Social Component:</code> Steers the particle towards the global best position.</li>
                  </ul>
            The particle's position is then updated by adding the new velocity to the current position.</li>
            <li><b>Iteration</b>: The process of evaluating, updating velocities, and updating positions is repeated for a specified number of generations. The algorithm converges when the global best position no longer changes significantly or when the maximum number of generations is reached.</li>
            <li><b>Result</b>: The algorithm returns the global best position (the set of weights and biases that minimize the error) and the corresponding error value.</li>
         </ul>
      </div>
      """, unsafe_allow_html=True)
    st.write(" ")
    exec(open("/mount/src/metaml/src/PSO.py").read())

with tab5:
    exec(open("/mount/src/metaml/src/performance.py").read())

with tab6:
    exec(open("/mount/src/metaml/src/chat.py").read())

with tab7:
   st.title("About Us")
   #  exec(open("/mount/src/metaml/src/aboutus.py").read())
