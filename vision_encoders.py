import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, TimeDistributed, Reshape, Concatenate, LSTM
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

import slicer_layer

# Функция для создания энкодера изображений на основе Xception
def create_xception_encoder(num_images_per_story, lstm_units):
    # Загрузка предобученной модели Xception для использования в качестве энкодера изображений
    xception_model = Xception(
        include_top=False, weights="imagenet", pooling="avg"
    )

    # Замораживание слоев модели Xception
    for layer in xception_model.layers:
        layer.trainable = False

    # Входной слой для изображений
    image_inputs = Input(shape=(num_images_per_story, 299, 299, 3), name="image_inputs")

    # Слои для преобразования и объединения эмбеддингов изображений
    reshaper = Reshape(target_shape=(1, xception_model.layers[-1].output_shape[-1]))
    concatenator = Concatenate(axis=1)
    image_embeddings = []

    # Получение изображений как входных данных и разбиение по временной оси
    sequential_inputs = slicer_layer.TimestepSliceLayer()(image_inputs)

    for img in sequential_inputs:
        # Применение модели Xception к каждому изображению
        preprocessed_img = preprocess_input(tf.cast(img, tf.float32))
        xception_features = xception_model(preprocessed_img)
        reshaped_features = reshaper(xception_features)
        image_embeddings.append(reshaped_features)

    lstm_input = concatenator(image_embeddings)

    # Применение LSTM для кодирования изображений, получение выхода размера (None, num_images_per_story, lstm_units)
    lstm_output = LSTM(lstm_units, return_sequences=True)(lstm_input)

    # Создание модели энкодера изображений
    return Model(inputs=image_inputs, outputs=lstm_output, name="xception_vision_encoder")
