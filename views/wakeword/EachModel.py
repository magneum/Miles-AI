import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model_files = os.listdir("temp/wakeword/models/")
models = []
histories = []
for model_file in model_files:
    if model_file.endswith(".h5"):  # Check for .h5 or .hdf5 file extension
        model = load_model(os.path.join("temp/wakeword/models/", model_file))
        models.append(model)
        history_file = model_file.replace(".h5", "_history.npy")
        history = np.load(
            os.path.join("temp/wakeword/models/", history_file), allow_pickle=True
        ).item()
        histories.append(history)

plt.figure(figsize=(10, 6))
for i, history in enumerate(histories):
    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    plt.plot(
        history.get(acc_key, history.get(val_acc_key)), label="Model {}".format(i + 1)
    )

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training Accuracy for Each Model")
plt.show()
