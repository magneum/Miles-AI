# Layer 1
model.add(Dense(units=units_per_layer[0], input_shape=(len(training_x[0]),))) # Add a dense layer with the specified number of units and input shape
model.add(Activation(activation_functions[0])) # Add the specified activation function
model.add(BatchNormalization()) # Add batch normalization for regularization
model.add(Dropout(0.5))  # Add a dropout layer with the specified rate for regularization

# Layer 2
model.add(Dense(units=units_per_layer[1])) # Add a dense layer with the specified number of units
model.add(Activation(activation_functions[1])) # Add the specified activation function
model.add(BatchNormalization()) # Add batch normalization for regularization
model.add(Dropout(0.5))  # Add a dropout layer with the specified rate for regularization

# Layer 3
model.add(Dense(units=units_per_layer[2])) # Add a dense layer with the specified number of units
model.add(Activation(activation_functions[2])) # Add the specified activation function
model.add(BatchNormalization()) # Add batch normalization for regularization
model.add(Dropout(0.5))  # Add a dropout layer with the specified rate for regularization

# Layer 4
model.add(Dense(units=units_per_layer[3])) # Add a dense layer with the specified number of units
model.add(Activation(activation_functions[3])) # Add the specified activation function
model.add(BatchNormalization()) # Add batch normalization for regularization
model.add(Dropout(0.5))  # Add a dropout layer with the specified rate for regularization

# Layer 5
model.add(Dense(units=units_per_layer[4])) # Add a dense layer with the specified number of units
model.add(Activation(activation_functions[4])) # Add the specified activation function
model.add(BatchNormalization()) # Add batch normalization for regularization
model.add(Dropout(0.5))  # Add a dropout layer with the specified rate for regularization

# Layer 6
model.add(Dense(units=units_per_layer[5])) # Add a dense layer with the specified number of units
model.add(Activation(activation_functions[5])) # Add the specified activation function
model.add(BatchNormalization()) # Add batch normalization for regularization
model.add(Dropout(0.5))  # Add a dropout layer with the specified rate for regularization

# Layer 7
model.add(Dense(units=units_per_layer[6])) # Add a dense layer with the specified number of units
model.add(Activation(activation_functions[6])) # Add the specified activation function
model.add(BatchNormalization()) # Add batch normalization for regularization
model.add(Dropout(0.5))  # Add a dropout layer with the specified rate for regularization

# Output layer
model.add(Dense(len(training_y[0]), activation="softmax")) # Add the output layer with softmax activation