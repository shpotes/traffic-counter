from typing import Tuple
from misc import prod

from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, Flatten, Reshape
from tensorflow.keras.models import Sequential, Model

def yolov1(input_shape: Tuple[int, ...]=(448, 448, 3), num_grid: int=7,
           bb_per_grid: int=2, num_classes: int=20) -> Model:
    output_shape = (num_grid, num_grid, bb_per_grid * 5 + num_classes)
    
    model = Sequential()
    # 1
    model.add(Conv2D(64, (7, 7), strides=2, input_shape=input_shape))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D((2, 2), strides=2))
    # 2
    model.add(Conv2D(192, (3, 3), strides=2)); model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D((2, 2), strides=2))
    # 3
    model.add(Conv2D(128, (1, 1))); model.add(LeakyReLU(0.1))
    model.add(Conv2D(256, (3, 3))); model.add(LeakyReLU(0.1))
    model.add(Conv2D(256, (1, 1))); model.add(LeakyReLU(0.1))
    model.add(Conv2D(512, (3, 3))); model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D((2, 2), strides=2))
    # 4
    for _ in range(4):
        model.add(Conv2D(256, (1, 1))); model.add(LeakyReLU(0.1))
        model.add(Conv2D(512, (3, 3))); model.add(LeakyReLU(0.1))
    model.add(Conv2D(512, (1, 1))); model.add(LeakyReLU(0.1))
    model.add(Conv2D(1024, (3, 3))); model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D((2, 2), strides=2))
    # 5
    for _ in range(2):
        model.add(Conv2D(512, (1, 1))); model.add(LeakyReLU(0.1))
        model.add(Conv2D(1024, (3, 3))); model.add(LeakyReLU(0.1))
    model.add(Conv2D(1024, (3, 3))); model.add(LeakyReLU(0.1))
    model.add(Conv2D(1024, (3, 3), strides=2)); model.add(LeakyReLU(0.1))
    # 6
    model.add(Conv2D(1024, (3, 3))); model.add(LeakyReLU(0.1))
    model.add(Conv2D(1024, (3, 3))); model.add(LeakyReLU(0.1))
    # Conn.
    model.add(Flatten())
    model.add(Dense(4096)); model.add(LeakyReLU(0.1))
    model.add(Dense(prod(output_shape)))
    model.add(Reshape(output_shape))

    return model
