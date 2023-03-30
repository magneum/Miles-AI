import os
import time
import json
import torch
import numpy as np
import torch.nn as nn
from termcolor import colored, cprint
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import NeuralNet
from utils import tokenize, stem, bag_of_words

# Load the intents file
with open("corpdata/intents.json", "r") as f:
    intents = json.load(f)

# Extract words, tags, and xy data from intents
all_words = []
tags = []
xy = []
ignore_words = ["?", "!", ".", ","]

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend([stem(word.lower()) for word in w if word not in ignore_words])
        xy.append((w, tag))

# Create X_train and Y_train arrays
X_train = []
Y_train = []
tag2idx = {tag: i for i, tag in enumerate(tags)}

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    Y_train.append(tag2idx[tag])

# Convert to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Define hyperparameters
batch_size = 33
hidden_size = 16
n_epochs = 10000
learning_rate = 0.001
output_size = len(tags)
input_size = len(X_train[0])

# Print input and output sizes and tags
print(colored(f"Input Size: {input_size}", "green"))
print(colored(f"Output Size: {output_size}, Tags: {tags}", "green"))
time.sleep(4)
os.system("clear")

# Create dataset and dataloader
class ChatDataset(Dataset):
    def __init__(self, X, Y):
        self.n_samples = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.n_samples


train_dataset = ChatDataset(X_train, Y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(1, n_epochs + 1):
    # Use tqdm to create a progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)
    for (words, labels) in progress_bar:
        words = words.to(device)
        labels = labels.to(device)
        outputs = model(words)
        loss = criterion(outputs, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar with the current loss
        progress_bar.set_postfix({"loss": loss.item()})

    # Print loss and accuracy every epochs
    if epoch % 100 == 0:
        with torch.no_grad():
            total, correct = 0, 0
            # Loop through the data loader to get predictions and calculate accuracy
            for i, (words, labels) in enumerate(train_loader):
                # Move data to device (GPU or CPU)
                words = words.to(device)
                labels = labels.to(device)
                outputs = model(words)
                # Get predicted labels by taking the argmax of the output scores
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                # Count the number of correct predictions
                correct += (predicted == labels).sum().item()

            # Calculate accuracy as the percentage of correct predictions
            accuracy = 100 * correct / total
            # Print epoch, loss, and accuracy with colored text
            print(
                f"\033[33mEpochs: \033[37m{int(epoch / 100)}/{int(n_epochs / 100)}\033[0m "
                f"\033[31mLoss: \033[37m{loss}\033[0m "
                f"\033[34mAccuracy: \033[37m{accuracy}\033[0m"
            )

# Print final loss and save the trained model
print(colored(f"Final-Loss: {loss.item():.4f}", "green"))

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags,
}
torch.save(data, "reviews/model.pth")

# Print message indicating training completion and model saving
print(colored(f"> Training completed <", "green"))
print(colored(f"> Model saved @model.pth <", "green"))
