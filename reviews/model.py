import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Define a function to load the pre-trained word embeddings
def load_embeddings(embeddings_file):
    # Load the pre-trained word embeddings from the file
    with open(embeddings_file, "r", encoding="utf-8") as f:
        word_embeddings = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            word_embeddings[word] = vector
    return word_embeddings


# Define a class to represent the dataset
class TextClassificationDataset(Dataset):
    def __init__(self, data, word_embeddings, max_sequence_length):
        self.data = data
        self.word_embeddings = word_embeddings
        self.max_sequence_length = max_sequence_length
        self.pad_token = np.zeros_like(next(iter(word_embeddings.values())))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text and label at the specified index
        text, label = self.data[idx]
        # Convert the text to a sequence of word embeddings
        sequence = []
        for word in text.split():
            if word in self.word_embeddings:
                sequence.append(self.word_embeddings[word])
            else:
                sequence.append(np.random.normal(scale=0.1, size=self.pad_token.shape))
        # Pad or truncate the sequence to the desired length
        if len(sequence) < self.max_sequence_length:
            sequence = np.concatenate(
                (
                    sequence,
                    np.tile(
                        self.pad_token, (self.max_sequence_length - len(sequence), 1)
                    ),
                )
            )
        else:
            sequence = sequence[: self.max_sequence_length]
        # Convert the sequence to a PyTorch tensor
        sequence = torch.from_numpy(np.array(sequence))
        # Convert the label to a PyTorch tensor
        label = torch.tensor(label)
        return sequence, label


# Define a class representing a neural network with three linear layers.
class NeuralNet(nn.Module):
    """
    A class representing a neural network with three linear layers.
    Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units in each hidden layer.
        output_size (int): The number of output classes.
    """

    # Define the constructor for the NeuralNet class
    def __init__(self, input_size, hidden_size, output_size):
        # Call the constructor of the nn.Module class
        super(NeuralNet, self).__init__()
        # Define the first linear layer with input_size input features and hidden_size output features
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Define the second linear layer with hidden_size input features and hidden_size output features
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Define the third linear layer with hidden_size input features and output_size output features
        self.fc3 = nn.Linear(hidden_size, output_size)
        # Define a ReLU activation function
        self.relu = nn.ReLU()

    # Define the forward function for the NeuralNet class
    def forward(self, x):
        # Pass the input tensor through the first linear layer
        out = self.fc1(x)
        # Apply the ReLU activation function to the output of the first linear layer
        out = self.relu(out)
        # Pass the output of the first linear layer through the second linear layer
        out = self.fc2(out)
        # Apply the ReLU activation function to the output of the second linear layer
        out = self.relu(out)
        # Pass the output of the second linear layer through the third linear layer
        out = self.fc3(out)
        # Return the output of the third linear layer as the final output of the neural network
        return out


# class NeuralNet(nn.Module):
#     """
#     A class representing a neural network with three linear layers.
#     Parameters:
#         input_size (int): The number of input features.
#         hidden_size (int): The number of hidden units in each hidden layer.
#         output_size (int): The number of output classes.
#     """

#     def __init__(self, input_size, hidden_size, output_size):
#         super(NeuralNet, self).__init__()

#         # Define the layers of the neural network
#         self.l1 = nn.Linear(input_size, hidden_size)  # First linear layer
#         self.bn1 = nn.BatchNorm1d(
#             hidden_size
#         )  # Batch normalization layer after first linear layer
#         self.dropout1 = nn.Dropout(0.5)  # Dropout layer after batch normalization layer
#         self.l2 = nn.Linear(hidden_size, hidden_size)  # Second linear layer
#         self.bn2 = nn.BatchNorm1d(
#             hidden_size
#         )  # Batch normalization layer after second linear layer
#         self.dropout2 = nn.Dropout(0.5)  # Dropout layer after batch normalization layer
#         self.l3 = nn.Linear(hidden_size, output_size)  # Output linear layer
#         self.relu = nn.ReLU()  # Rectified linear unit activation function

#     def forward(self, x):
#         """
#         Perform a forward pass through the neural network.

#         Parameters:
#             x (torch.Tensor): The input tensor of shape (batch_size, input_size).

#         Returns:
#             torch.Tensor: The output tensor of shape (batch_size, output_size).
#         """
#         # Perform the forward pass through each layer of the network, applying the ReLU activation function to each hidden layer
#         out = self.l1(
#             x
#         )  # Sure, here are the comments for each line of the forward method:
#         # Pass the input tensor through the first linear layer.
#         out = self.bn1(out)
#         # Apply batch normalization to the output of the first linear layer.
#         out = self.relu(out)
#         # Apply the ReLU activation function to the output of the first batch normalization layer.
#         out = self.dropout1(out)
#         # Apply dropout regularization to the output of the first ReLU layer.
#         out = self.l2(out)
#         # Pass the output of the first dropout layer through the second linear layer.
#         out = self.bn2(out)
#         # Apply batch normalization to the output of the second linear layer.
#         out = self.relu(out)
#         # Apply the ReLU activation function to the output of the second batch normalization layer.
#         out = self.dropout2(out)
#         # Apply dropout regularization to the output of the second ReLU layer.
#         out = self.l3(out)
#         # Pass the output of the second dropout layer through the final linear layer, which produces the output tensor.
#         return out
