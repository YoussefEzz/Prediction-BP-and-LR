import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
    weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
    return weights_input_hidden, weights_hidden_output

def forward_propagation(inputs, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    return hidden_layer_output, predicted_output

def backward_propagation(inputs, targets, hidden_layer_output, predicted_output, weights_hidden_output):
    output_error = targets - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    return output_delta, hidden_layer_delta

def update_weights(inputs, hidden_layer_output, output_delta, hidden_layer_delta, weights_input_hidden, weights_hidden_output, learning_rate):
    weights_hidden_output +=np.outer(hidden_layer_output,output_delta) * learning_rate
    weights_input_hidden += np.outer(inputs,hidden_layer_delta) * learning_rate

def calculate_error(targets, predicted_output):
    return np.mean((targets - predicted_output) ** 2)

def train_neural_network(inputs, targets, validation_inputs, validation_targets, hidden_size, epochs, learning_rate):
    input_size = inputs.shape[1]
    output_size = targets.shape[1]

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    training_errors = []
    validation_errors = []

    for epoch in range(epochs):
        for pattern in range(len(inputs)):
            random_index = np.random.randint(len(inputs))
            current_input = inputs[random_index]
            current_target = targets[random_index]

            hidden_layer_output, predicted_output = forward_propagation(current_input, weights_input_hidden, weights_hidden_output)

            output_delta, hidden_layer_delta = backward_propagation(current_input, current_target, hidden_layer_output, predicted_output, weights_hidden_output)

            update_weights(current_input, hidden_layer_output, output_delta, hidden_layer_delta, weights_input_hidden, weights_hidden_output, learning_rate)

        _, training_predicted_output = forward_propagation(inputs, weights_input_hidden, weights_hidden_output)
        training_error = calculate_error(targets, training_predicted_output)
        training_errors.append(training_error)

        _, validation_predicted_output = forward_propagation(validation_inputs, weights_input_hidden, weights_hidden_output)
        validation_error = calculate_error(validation_targets, validation_predicted_output)
        validation_errors.append(validation_error)

    # Plot the training and validation errors

    plt.plot(np.arange(1, epochs + 1), training_errors, label='Training Error')
    plt.plot(np.arange(1, epochs+ 1), validation_errors, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Quadratic Error')
    plt.legend()
    plt.show()

    plt.waitforbuttonpress()

    return weights_input_hidden, weights_hidden_output

# Example usage:
# Replace the following arrays with your data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])
validation_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
validation_targets = np.array([[0], [1], [1], [0]])

hidden_size = 4
epochs = 10000
learning_rate = 0.1

trained_weights_input_hidden, trained_weights_hidden_output = train_neural_network(inputs, targets, validation_inputs, validation_targets, hidden_size, epochs, learning_rate)

# Test the trained network
test_input = np.array([[0, 0]])
_, predicted_output = forward_propagation(test_input, trained_weights_input_hidden, trained_weights_hidden_output)
print(predicted_output)