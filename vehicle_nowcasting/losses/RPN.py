import tensorflow as tf

class SmoothL1(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        cond = tf.abs(error) < 1
        squared_loss = 0.5 * tf.square(loss)
        linear_loss = tf.abs(error) - 0.5
        return tf.where(cond, squared_loss, linear_loss)
