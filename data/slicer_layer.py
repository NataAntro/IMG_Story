import tensorflow as tf
from tensorflow.keras.layers import Layer

# Класс для разбиения тензора по временной оси
class TimestepUnstackLayer(Layer):
    def __init__(self, **kwargs):
        super(TimestepUnstackLayer, self).__init__(**kwargs)

    # Метод call для разбиения входных данных по оси 1
    def call(self, inputs, **kwargs):
        return tf.unstack(inputs, axis=1)

    # Метод для получения конфигурации слоя
    def get_config(self):
        return super(TimestepUnstackLayer, self).get_config()
