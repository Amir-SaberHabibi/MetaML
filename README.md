
![Project Logo](src/images/logo.png)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-0.86%2B-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

This project is designed to train neural networks using different optimization algorithms (Backpropagation, Genetic Algorithm, and Particle Swarm Optimization) and compare their results. The app provides an interface space to experiment with these algorithms and visualize their performance.

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
