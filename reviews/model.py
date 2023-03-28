import torch.nn as nn


class NeuralNet(nn.Module):
    """
    A class representing a neural network with three linear layers.
    Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units in each hidden layer.
        output_size (int): The number of output classes.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()

        # Define the layers of the neural network
        self.l1 = nn.Linear(input_size, hidden_size)  # First linear layer
        self.bn1 = nn.BatchNorm1d(
            hidden_size
        )  # Batch normalization layer after first linear layer
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer after batch normalization layer
        self.l2 = nn.Linear(hidden_size, hidden_size)  # Second linear layer
        self.bn2 = nn.BatchNorm1d(
            hidden_size
        )  # Batch normalization layer after second linear layer
        self.dropout2 = nn.Dropout(0.5)  # Dropout layer after batch normalization layer
        self.l3 = nn.Linear(hidden_size, output_size)  # Output linear layer
        self.relu = nn.ReLU()  # Rectified linear unit activation function

    def forward(self, x):
        """
        Perform a forward pass through the neural network.

        Parameters:
            x (torch.Tensor): The input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_size).
        """
        # Perform the forward pass through each layer of the network, applying the ReLU activation function to each hidden layer
        out = self.l1(
            x
        )  # Sure, here are the comments for each line of the forward method:
        # Pass the input tensor through the first linear layer.
        out = self.bn1(out)
        # Apply batch normalization to the output of the first linear layer.
        out = self.relu(out)
        # Apply the ReLU activation function to the output of the first batch normalization layer.
        out = self.dropout1(out)
        # Apply dropout regularization to the output of the first ReLU layer.
        out = self.l2(out)
        # Pass the output of the first dropout layer through the second linear layer.
        out = self.bn2(out)
        # Apply batch normalization to the output of the second linear layer.
        out = self.relu(out)
        # Apply the ReLU activation function to the output of the second batch normalization layer.
        out = self.dropout2(out)
        # Apply dropout regularization to the output of the second ReLU layer.
        out = self.l3(out)
        # Pass the output of the second dropout layer through the final linear layer, which produces the output tensor.
        return out
