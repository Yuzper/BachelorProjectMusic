import os
import math
import shutil
import numpy as np
import pandas as pd
from math import floor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, GRU, TimeDistributed, BatchNormalization, Rescaling
from tensorflow.keras.models import load_model
import tensorcross

tf.random.set_seed(42)
np.random.seed(42)


import keras.backend as K

def macro_f1_score(y_true, y_pred):
    """
    Calculate macro-average F1 score using Keras backend.

    Parameters:
        y_true (tensor or array): The true class labels (ground truth) as a tensor or Numpy array.
        y_pred (tensor or array): The predicted class labels as a tensor or Numpy array.

    Returns:
        float: The macro-average F1 score.
    """
    # Convert Numpy arrays to Keras backend tensors
    if not tf.is_tensor(y_true):
        y_true = K.constant(y_true)

    if not tf.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    
    # Calculate true positives, false positives, and false negatives for each class
    tp = K.sum(y_true * K.round(y_pred), axis=0)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=0)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=0)

    # Calculate precision and recall for each class
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    # Calculate F1 score for each class
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())

    # Calculate macro-average F1 score
    macro_f1 = K.mean(f1_score)

    return macro_f1


### CRNN Model ###
# Defines function for model artitecture: REMOVED GRU_units!!!!!!!!!!!!!!!!
def create_model(n_filters, kernel_sizes, pool_sizes, GRU_units, dropout_rate, learning_rate, epochs, batch_size): # 8 Hyperparameters
    # Define the input shape
    input_shape = (235, 352, 3)

    # Define the CRNN architecture
    model = Sequential()
    model.add(Rescaling(1./255, input_shape = input_shape))

    # Conv Block 1
    model.add(Conv2D(filters=n_filters[3], kernel_size=kernel_sizes, activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=n_filters[3], kernel_size=kernel_sizes, activation="relu"))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=pool_sizes))

    # Conv Block 2
    model.add(Conv2D(filters=n_filters[2], kernel_size=kernel_sizes, activation="relu"))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=n_filters[2], kernel_size=kernel_sizes, activation="relu"))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=pool_sizes))    

    # Conv Block 3
    model.add(Conv2D(filters=n_filters[1], kernel_size=kernel_sizes, activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=n_filters[1], kernel_size=kernel_sizes, activation="relu"))
    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=pool_sizes))    

    # Conv Block 4
    model.add(Conv2D(filters=n_filters[0], kernel_size=kernel_sizes, activation="relu"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=n_filters[0], kernel_size=kernel_sizes, activation="relu"))
    model.add(BatchNormalization())

    # Recurrent layers
    model.add(TimeDistributed(Flatten()))
    #model.add(GRU(GRU_units*2, return_sequences=True))
    model.add(GRU(GRU_units, return_sequences=False))
    model.add(Dropout(dropout_rate))

    # Flatten
    #model.add(Flatten())    

    # Fully connected layers
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))
   
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(dropout_rate))
 
    num_classes = 5
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy", macro_f1_score]
                 )
    return model

### Create Dataset ###
batch_size = 64

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
trainingData = tf.keras.utils.image_dataset_from_directory(
    directory = "SpectrogramData/training",
    labels="inferred",
    label_mode = "categorical",
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(235, 352),
    shuffle=True,
    seed=42
)

print()

# Construct dataset for best model, making a seperate dataset since I need a random split of validation here.
bestTrainingData = tf.keras.utils.image_dataset_from_directory(
    directory = "SpectrogramData/training",
    labels="inferred",
    label_mode = "categorical",
    color_mode="rgb",
    validation_split=0.2,
    subset="both",
    batch_size=batch_size,
    image_size=(235, 352),
    shuffle=True,
    seed=42
)

### Training ###
from tensorcross.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    "n_filters": [(128,64,32,16), (16,32,64,128), (256,128,64,32)],
    "kernel_sizes": [(3,3), (4,4)],
    "pool_sizes": [(2,2), (3,3)],
    "GRU_units": [128, 256],
    "dropout_rate": [0.2, 0.3],
    "learning_rate": [0.01, 0.0001, 0.000001],
    "epochs": [200],
    "batch_size": [64]
}

# 144 combinations

kFolds = 5
grid_search_cv = GridSearchCV(model_fn = create_model,
                          param_grid = param_grid,
                          n_folds = kFolds,
                          verbose = 1)

grid_search_cv.fit(dataset = trainingData,
		callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
		verbose = 1)

best_params_ = grid_search_cv.results_["best_params"]

# Retraining best model
# Create the model with the current parameter values
bestModelArchitecture = create_model(n_filters = best_params_["n_filters"],
                     kernel_sizes = best_params_["kernel_sizes"],
                     pool_sizes = best_params_["pool_sizes"],
                     GRU_units = best_params_["GRU_units"],
                     dropout_rate = best_params_["dropout_rate"],
                     learning_rate = best_params_["learning_rate"],
                     epochs = best_params_["epochs"],
                     batch_size = best_params_["batch_size"])

# Train and evaluate the model
bestModel = bestModelArchitecture.fit(bestTrainingData[0],
		validation_data = bestTrainingData[1],
		epochs=best_params_["epochs"],
		callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
		verbose=2)


print()
print("######################################################\n")
print("Baseline")
print("Loss")
print(bestModel.history["loss"],"\n")
print("Accuracy")
print(bestModel.history["accuracy"],"\n")
print("Macro f1 Score")
if "macro_f1_score" in bestModel.history:
    print(bestModel.history["macro_f1_score"])
else:
    print("Macro f1 score not available in history.")
print(grid_search_cv.results_["best_params"])


import csv

# Combine the lists into rows
rows = zip(bestModel.history['accuracy'],
           bestModel.history['val_accuracy'],
           bestModel.history['macro_f1_score'],
           bestModel.history['val_macro_f1_score'],
           bestModel.history['loss'],
           bestModel.history['val_loss']
          )

# Specify the CSV file name
csv_file = "Results/Baseline_metrics.csv"

# Write the rows to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['accuracy','val_accuracy','macro_f1_score','val_macro_f1_score','loss','val_loss'])  # Write header
    writer.writerows(rows)
    

bestModelArchitecture.save("Results/SavedModels/best_model_baseline.h5")




