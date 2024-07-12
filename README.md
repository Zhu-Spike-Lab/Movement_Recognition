# Comparative Analysis of Evolutionary Algorithm (EA) and BackPropagation (BP) in Moving Digit Pattern Recognition

This repository contains an implementation of a Recurrent Spiking Neural Network (RSNN) designed to classify directional movements using an evolutionary algorithm & backpropagation through time for optimization.

## Table of Contents

- [Setup Environment](#setup-environment)
- [Access Training Data](#access-training-data)
- [Save and Access Data](#save-and-access-data)
- [Script and Function Descriptions](#script-and-function-descriptions)
- [Processes and Algorithms](#processes-and-algorithms)
- [Analysis and Plotting](#analysis-and-plotting)
- [Future Experiments](#future-experiments)
- [Common Issues and Solutions](#common-issues-and-solutions)


## Setup Environment

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Zhu-Spike-Lab/Movement_Recognition
    cd Movement_Recognition
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Access Training Data

1. **Data File**: The training data is provided in the `movement_sequences.csv` file.
2. **Load Data**: The data can be loaded using pandas.
    ```python
    import pandas as pd
    df = pd.read_csv('movement_sequences.csv')
    ```

## Save and Access Data

1. **Saving Models**: Models can be saved using `torch.save`.
    ```python
    torch.save(model.state_dict(), "best_model.pth")
    ```

2. **Loading Models**:
    ```python
    model = RSNN_direction()
    model.load_state_dict(torch.load("best_model.pth"))
    ```

## Script and Function Descriptions

### `directions.ipynb`

- **Purpose**: Contains code for setting up and making the dataset for the moving digit task.
- **Key Functions**:
  - `generate_movement_sequence`: generate the moving digit frame sequence
  - `generate_all_sequences`: generate all possible sequences

### `engine.ipynb`

- **Purpose**: Implements the evolutionary algorithm to optimize the RSNN model.
- **Key Functions**:
  - `encode_model(model)`: Encodes model parameters into a gene.
  - `decode_model(model, gene)`: Decodes a gene back into model parameters.
  - `Evolution`: Class containing methods for population initialization, evaluation, selection, crossover, mutation, and evolution.
  - `evolve(n_models, n_offspring, n_generations, dataloader, mutation_rate)`: Runs the evolutionary algorithm over multiple generations.

### `train_moving_digit_evol.ipynb`

- **Purpose**: Implements the BackProp Through Time to optimize the RSNN model.
 

## Analysis and Plotting

### Fitness and Weight Analysis

- **Plot Best Fitness**: Track the best fitness score over generations.
    ```python
    def plot_best_fitness(all_best_fitness):
        plt.plot(all_best_fitness)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness (Loss)")
        plt.title("Evolution of Best Fitness")
        plt.show()
    ```

- **Plot Average Fitness**: Track the average fitness score over generations.
    ```python
    def plot_average_fitness(all_fitness):
        average_fitness = [np.mean(fitness) for fitness in all_fitness]
        plt.plot(average_fitness)
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness (Loss)")
        plt.title("Evolution of Average Fitness")
        plt.show()
    ```

## Future Experiments

1. **Parameter Tuning**: Experiment with different values for `beta`, `pe_e`, and mutation rates to improve model performance.
2. **Additional Metrics**: Incorporate more metrics like criticality and synchrony into the fitness function.
3. **Extended Data**: Use larger and more varied datasets to test the model's robustness.

## Common Issues and Solutions

1. **High Initial Loss**:
    - Ensure data normalization.
    - Verify model initialization.

2. **Vanishing/Exploding Gradients**:
    - Untrain the input layer and recurrent layer

