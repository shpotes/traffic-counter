#!/usr/bin/env python
from typing import Tuple

from tensorflow.keras.models import Model, Sequential, Flatten
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D

def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    """
    Return LeNet Keras model
    
    Just another starter model, useful for image recognition tasks
    """
    
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu',
               input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')])

    return model
