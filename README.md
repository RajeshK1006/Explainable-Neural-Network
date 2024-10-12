

# Explainable Neural Network (MINI-Project)

This project implements a feedforward neural network for MNIST digit classification, enhanced with Explainable AI techniques like saliency maps and activation visualizations to increase model interpretability and trust.


## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
  - [Predictions](#predictions)
  - [Saliency Map](#saliency-map)
  - [Activations](#activations)
- [Performance Plot](#performance-plot)
- [Usage](#usage)
- [License](#license)

## Introduction
This project implements a simple feedforward neural network from scratch to classify handwritten digits (0-9) using the MNIST dataset. While neural networks are powerful tools for pattern recognition and classification, they are often viewed as "black boxes" due to their complex internal workings, which can make it difficult to understand why they make certain predictions.

To address this, we incorporate Explainable AI (XAI) techniques like saliency maps and activation visualizations. These methods help us interpret the neural network's decision-making process, increasing trust in its predictions by transforming it from a "black box" into a more transparent or "white box" model.

## How XAI Techniques Are Used:
# Saliency Maps:

Saliency maps allow us to visualize which pixels in an input image had the most influence on the network’s decision. By highlighting important features, they provide insight into the network’s focus areas, helping us understand why the model predicts a certain class. For example, when classifying a digit, the saliency map shows which parts of the image the model found most relevant, enabling us to verify whether the network is focusing on the correct regions of the input.

# Activation Visualizations:

By visualizing the activations of neurons in each layer, we gain insight into how the input data is transformed as it moves through the network. This step-by-step view of the neural network’s processing helps us understand how intermediate representations are built and how the final prediction is derived. Each layer’s activations offer a glimpse into the features the network is learning, from simple edges in early layers to more complex shapes in deeper layers.
# Purpose and Benefits:
The goal of these XAI techniques is to build user trust in the model by making its inner workings interpretable. With saliency maps, we can ensure that the network is focusing on appropriate parts of the image, and with activation visualizations, we can follow how the model processes information. This level of interpretability transforms the model into a more understandable and trustworthy system, improving user confidence in its predictions.

By providing these interpretability tools, the project not only aims to build an effective model for digit classification but also highlights the importance of transparency in machine learning models, especially when applied to real-world decision-making systems.


## Requirements
Ensure you have the following Python libraries installed:
- `numpy`
- `matplotlib`
- `sklearn`

You can install the required libraries using:
```bash
pip install numpy matplotlib scikit-learn
```

## Project Structure
- `mnist_nn.py`: Main file that defines the neural network, training loop, and visualization functions.
- `README.md`: Documentation file.
- `data/`: Directory containing any saved data or model weights (if applicable).

## Model Architecture
The neural network consists of:
- **Input Layer**: 784 units (28x28 flattened images).
- **Hidden Layers**: 2 fully connected layers with ReLU activation.
  - First hidden layer: 32 neurons
  - Second hidden layer: 32 neurons
- **Output Layer**: 10 units with softmax activation for digit classification (0-9).

### Weights Initialization
The model uses Xavier initialization for better weight distribution during training.

### Activation Functions
- Hidden layers use ReLU.
- Output layer uses Softmax for classification.

## Training
The model is trained on 80% of the MNIST dataset, with 20% used for testing. During training, the network updates its weights using backpropagation and gradient descent.

Key features of training:
- **Loss Functions**: 
  - Cross-Entropy Loss
  - Sum Squared Residuals (SSR)
- **Optimization**: Gradient Descent
- **Learning Rate**: 0.01
- **Epochs**: 1000

## Evaluation
The model's performance is evaluated on the test set using accuracy as the primary metric. Additionally, cross-entropy loss and SSR loss are computed during training.

## Visualizations
The project provides multiple visualizations to interpret the model's predictions, activations, and saliency maps.

### Predictions
You can visualize random samples from the test set with their true labels and predicted labels.
```python
nn.visualize_predictions(X_test, y_test, num_samples=10)
```

### Saliency Map
Generate a saliency map to understand which parts of the input image contributed most to a particular class prediction.
```python
saliency = saliency_map(nn, X_test[0:1], target_class_index)
visualize_saliency(saliency, X_test[0])
```

### Activations
Visualize activations of the input, hidden, and output layers for a given test sample.
```python
sample_image = X_test[0].reshape(1, -1)
activations = visualize_activations(nn, sample_image)
```

## Performance Plot
The model tracks cross-entropy loss, SSR loss, and accuracy over epochs. You can plot these metrics for better insight into the model's performance during training.
```python
plot_model_performance(metrics)
```

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/RajeshK1006/Explainable-Neural-Network.git
    cd mnist-nn
    ```
2. Run the script to train the model:
    ```bash
    python mnist_nn.ipynb
    ```
3. Modify hyperparameters such as learning rate, number of epochs, and layer sizes in the script as needed.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
```

This README is designed to give users an overview of your project, how it works, and how they can use and modify it. Let me know if you'd like to make any adjustments!
