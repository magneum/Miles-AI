# This is a convolutional neural network (CNN) model for facial expression recognition.

# These are the necessary imports for TensorFlow and Keras libraries to be used in the code.
import tensorflow
from tensorflow.python.keras import layers
from keras.applications import ImageDataGenerator

# This is the CNN model architecture defined using Keras Sequential API.
# The model contains four sets of Convolutional + Batch Normalization + MaxPooling + Dropout layers, followed by three fully connected (Dense) layers with Batch Normalization and Dropout.
# The output layer has 7 nodes with a softmax activation function for multiclass classification.
Facemodel = tensorflow.keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(512, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(1024, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(7, activation="softmax"),
    ]
)

# This compiles the model with the Adam optimizer, categorical cross-entropy as the loss function, and accuracy as the metric to be used during training.
Facemodel.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# This sets up the ImageDataGenerator to load and preprocess the training and test datasets.
# The training dataset undergoes rescaling, shearing, zooming, and horizontal flipping.
# Both datasets are loaded in grayscale mode with a target size of 48x48 and a batch size of 32. Class mode is set to categorical for multiclass classification.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
)

validation_generator = test_datagen.flow_from_directory(
    "data/test",
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
)


# This trains the model using the training dataset with a defined number of epochs, batch size, and validation set.
# The training is performed using the fit method.
Facemodel.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
)

# This evaluates the model performance on the test dataset
test_loss, test_acc = Facemodel.evaluate(
    validation_generator, steps=len(validation_generator)
)
print("Test accuracy:", test_acc)


# Save the model into .h5 format which can be used later
Facemodel.save("face_expression_model.h5")
