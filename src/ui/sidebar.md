# MetaML

This project is a Streamlit application designed to train neural networks using different optimization algorithms (Backpropagation, Genetic Algorithm, and Particle Swarm Optimization) and compare their results. The app provides a user-friendly interface to experiment with these algorithms and visualize their performance.

## Project Structure

The project consists of the following key components:

- **Training Algorithms**: Implements **Backpropagation, Genetic Algorithm, and Particle Swarm Optimization** for training neural networks.
- **Comparison Module**: Compares the **performance** of the different algorithms based on predefined metrics and generates visualizations.
- **Data Analyzer Chatbot**: Utilizes the power of **Llama3 large language model** to analyze the performance of different algorithms (PSO, BP, GA) and provide insights through a chatbot interface.
- **Resources**: Includes **data files** for storing results in CSV format.


## How to Run the App

1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run home.py
    ```

3. Navigate through the app using the sidebar to select algorithms, upload datasets, and view results.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to create a pull request or open an issue.

## License

This project is licensed under the MIT License.
