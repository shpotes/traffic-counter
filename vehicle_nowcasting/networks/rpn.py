from tensorflow.keras.layers import Conv2D, Concatenate

def RPN(inputs, k):
    x = Conv2D(256, kernel_size=(3, 3),
               strides=1, activation='relu')(inputs)
    cls = Conv2D(2 * k, kernel_size=(1, 1))(x)
    reg = Conv2D(4 * k, kernel_size=(1, 1))(x)
    return Concatenate()([cls, reg])
