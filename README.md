# Neural Network Predictor

This repository contains a basic 2-layer Artificial Neural Network (ANN) implemented using Numpy in Python. The ANN is designed to predict the next number in a given series, demonstrating fundamental concepts of feedforward networks, activation functions, and backpropagation.

## Features

- **Input Layer**: Takes input data as a matrix and passes it on to the hidden layers.
- **Hidden Layers**: Three hidden layers utilizing ReLU activation functions to introduce non-linearity.
- **Output Layer**: Returns the predicted output based on the processed input.
- **Error Calculation**: Calculates the error between the predicted output and the expected output using Mean Squared Error (MSE).
- **Weight Updates**: Updates weights and biases using backpropagation with gradient descent.
- **Training**: The network is trained over multiple epochs (up to 30,000) to minimize the prediction error.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/neural-network-predictor.git
   
2. Navigate to the repository directory:
   cd neural-network-predictor

    
3. Run the Python script:
   python your-script-name.py

