import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Build an NN to convert Celsius to Fahrenheit by discovering the formula  f = 1.8c +32
celsius_q = np.array([-40, -10, 0, 8 ,15, 22, 38], dtype = float) # FEATURES Input
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float) # LABELS Wanted Output to be trained

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit" .format(c, fahrenheit_a[i]))

# Set up a simple, Dense Neural network with only ONE Layer and a single neuron
L0 = tf.keras.layers.Dense(units = 1, input_shape = [1])
# Layer 0 , input_shape[1] = 1D array with ONE member, units = number of neurons
# Executes the equation a1 = x1 * w11 + b1 / Output = Input * Weight + Bias

# Put the layer in the model
model = tf.keras.Sequential([L0])

# Compile model with loss and optimizer functions with the respective learning rate
model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))

# Train the model using Gradient Descent
history = model.fit(celsius_q, fahrenheit_a, epochs = 500, verbose = False) # Model with 7 pairs over 500 epochs = 3500 examples
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])  # The more training the better, Loss converges to zero 
plt.show()

# Predictions
print(model.predict([100.0])) # Correct answer is 212, will give us 211.32118

# Layer Weights, we take 1.822 and 29.08. Remember the actual equation f = 1.8c +32
print("Theese are the layer 0 (L0) variables: {}" .format(L0.get_weights()))