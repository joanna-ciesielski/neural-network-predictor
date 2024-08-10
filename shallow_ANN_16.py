import numpy as np

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Normalize function
def normalize(data, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        min_val = np.min(data)
        max_val = np.max(data)
    return (data - min_val) / (max_val - min_val), min_val, max_val

# Denormalize function
def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

# Expanded training data: more sequences of numbers
X = np.array([
    [0, 1, 2],
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10],
    [9, 10, 11],
    [10, 11, 12],
    [11, 12, 13],
    [12, 13, 14],
    [13, 14, 15],
    [14, 15, 16],
    [15, 16, 17]
])
y = np.array([
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
    [11],
    [12],
    [13],
    [14],
    [15],
    [16],
    [17],
    [18]
])

# Normalize training data
X_norm, X_min, X_max = normalize(X)
y_norm, y_min, y_max = normalize(y)

# Xavier initialization
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = np.sqrt(2 / (in_dim + size[1]))
    return np.random.randn(*size) * xavier_stddev

# Initialize weights and biases with Xavier initialization
input_layer_neurons = X_norm.shape[1]
hidden_layer_neurons1 = 80  # Number of neurons in the first hidden layer
hidden_layer_neurons2 = 80  # Number of neurons in the second hidden layer
hidden_layer_neurons3 = 80  # Number of neurons in the third hidden layer
output_neurons = 1

np.random.seed(42)
weights_input_hidden1 = xavier_init((input_layer_neurons, hidden_layer_neurons1))
weights_hidden1_hidden2 = xavier_init((hidden_layer_neurons1, hidden_layer_neurons2))
weights_hidden2_hidden3 = xavier_init((hidden_layer_neurons2, hidden_layer_neurons3))
weights_hidden3_output = xavier_init((hidden_layer_neurons3, output_neurons))
bias_hidden1 = np.zeros((1, hidden_layer_neurons1))
bias_hidden2 = np.zeros((1, hidden_layer_neurons2))
bias_hidden3 = np.zeros((1, hidden_layer_neurons3))
bias_output = np.zeros((1, output_neurons))

# Training the ANN
epochs = 30000  # Number of epochs
initial_learning_rate = 0.001  # Learning rate
patience = 5000  # Early stopping patience

best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    learning_rate = initial_learning_rate * (1 / (1 + 0.0005 * epoch))  # Adjusted decay learning rate
    
    # Feedforward
    hidden_layer_input1 = np.dot(X_norm, weights_input_hidden1) + bias_hidden1
    hidden_layer_output1 = relu(hidden_layer_input1)
    
    hidden_layer_input2 = np.dot(hidden_layer_output1, weights_hidden1_hidden2) + bias_hidden2
    hidden_layer_output2 = relu(hidden_layer_input2)
    
    hidden_layer_input3 = np.dot(hidden_layer_output2, weights_hidden2_hidden3) + bias_hidden3
    hidden_layer_output3 = relu(hidden_layer_input3)
    
    output_layer_input = np.dot(hidden_layer_output3, weights_hidden3_output) + bias_output
    predicted_output = output_layer_input  # Linear activation for regression
    
    # Backpropagation
    error = y_norm - predicted_output
    d_predicted_output = error  # Derivative of linear activation is 1
    
    error_hidden_layer3 = d_predicted_output.dot(weights_hidden3_output.T)
    d_hidden_layer3 = error_hidden_layer3 * relu_derivative(hidden_layer_output3)
    
    error_hidden_layer2 = d_hidden_layer3.dot(weights_hidden2_hidden3.T)
    d_hidden_layer2 = error_hidden_layer2 * relu_derivative(hidden_layer_output2)
    
    error_hidden_layer1 = d_hidden_layer2.dot(weights_hidden1_hidden2.T)
    d_hidden_layer1 = error_hidden_layer1 * relu_derivative(hidden_layer_output1)
    
    # Gradient clipping
    max_gradient_norm = 1.0
    d_hidden_layer1 = np.clip(d_hidden_layer1, -max_gradient_norm, max_gradient_norm)
    d_hidden_layer2 = np.clip(d_hidden_layer2, -max_gradient_norm, max_gradient_norm)
    d_hidden_layer3 = np.clip(d_hidden_layer3, -max_gradient_norm, max_gradient_norm)
    d_predicted_output = np.clip(d_predicted_output, -max_gradient_norm, max_gradient_norm)
    
    # Updating Weights and Biases
    weights_hidden3_output += hidden_layer_output3.T.dot(d_predicted_output) * learning_rate
    weights_hidden2_hidden3 += hidden_layer_output2.T.dot(d_hidden_layer3) * learning_rate
    weights_hidden1_hidden2 += hidden_layer_output1.T.dot(d_hidden_layer2) * learning_rate
    weights_input_hidden1 += X_norm.T.dot(d_hidden_layer1) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden3 += np.sum(d_hidden_layer3, axis=0, keepdims=True) * learning_rate
    bias_hidden2 += np.sum(d_hidden_layer2, axis=0, keepdims=True) * learning_rate
    bias_hidden1 += np.sum(d_hidden_layer1, axis=0, keepdims=True) * learning_rate
    
    # Calculate and print the loss
    loss = mse_loss(y_norm, predicted_output)
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')
    
    # Early stopping
    if loss < best_loss:
        best_loss = loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter > patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Testing the ANN with a new sequence
test_input = np.array([7,8,9])
test_input_norm, _, _ = normalize(test_input, X_min, X_max)  # Normalize test input with training data's min and max
hidden_layer_input1 = np.dot(test_input_norm, weights_input_hidden1) + bias_hidden1
hidden_layer_output1 = relu(hidden_layer_input1)
hidden_layer_input2 = np.dot(hidden_layer_output1, weights_hidden1_hidden2) + bias_hidden2
hidden_layer_output2 = relu(hidden_layer_input2)
hidden_layer_input3 = np.dot(hidden_layer_output2, weights_hidden2_hidden3) + bias_hidden3
hidden_layer_output3 = relu(hidden_layer_input3)
output_layer_input = np.dot(hidden_layer_output3, weights_hidden3_output) + bias_output
predicted_output = output_layer_input  # Linear activation for regression

# Denormalize the output
predicted_output = denormalize(predicted_output, y_min, y_max)

print(f'Predicted next number in the sequence [7,8,9] is: {predicted_output[0]}')
