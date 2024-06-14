import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Concatenate, Reshape, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Количество изображений в одной истории
num_photos_per_story = 5

# Класс кастомной модели
class CustomModel(Model):
    def __init__(self, vision_encoder, text_encoder, decoder, word_index, words_per_caption, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.decoder = decoder

        self.reshape_loss = Reshape(target_shape=(5, words_per_caption))
        self.reshape_image = Reshape(target_shape=(1, 299, 299, 3))
        self.reshape_caption = Reshape(target_shape=(1, words_per_caption))
        self.concat_sequence = Concatenate(axis=1)
        self.vectorize_text = TextVectorization(
            max_tokens=None,
            standardize=None,                           # Входные данные уже стандартизированы
            output_mode='int',
            output_sequence_length=words_per_caption,
            vocabulary=list(word_index.keys())[1:]    # Исключение токена <OOV>, который добавляется автоматически
        )

        self.loss_tracker = Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False, **kwargs):
        true_labels = []
        images = []

        for i in range(num_photos_per_story):
            # Изменение формы изображения до (None, 1, 299, 299, 3) для последующего объединения
            reshaped_img = self.reshape_image(inputs['image_' + str(i)])
            images.append(reshaped_img)

            # Векторизация правильной подписи к изображению => (None, words_per_caption)
            correct_caption = self.vectorize_text(inputs['caption_' + str(i)])

            # Изменение формы правильной подписи до (None, 1, words_per_caption)
            reshaped_caption = self.reshape_caption(correct_caption)
            true_labels.append(reshaped_caption)

        # Объединение всех правильных подписей для формирования тензора формы (None, 5, words_per_caption)
        true_text_indices = self.concat_sequence(true_labels)

        # Объединение изображений для получения тензора формы (None, 5, 299, 299, 3)
        image_sequence = self.concat_sequence(images)

        # Кодирование изображений => (None, 5, image_encoder_lstm_size)
        encoded_images = self.vision_encoder(image_sequence)

        # Применение декодера
        predicted_text_softmax = self.decoder(encoded_images)

        # (None, 5, words_per_caption), (None, 5, words_per_caption, vocab_len)
        return true_text_indices, predicted_text_softmax

    def compute_loss(self, true_text, predicted_text):
        embedding_dist_weight = 0.2
        # Вычисление категориальной кросс-энтропии между предсказанными и правильными подписями
        # Используем sparse, так как используем индексы слов, а не one-hot кодирование.
        # use_logits=False, так как предсказанный текст представляет собой активации softmax
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=true_text, y_pred=predicted_text
        )

        # y_pred и y_true формы (None, 5, words_per_caption)
        true_text_indices = tf.keras.backend.cast(true_text, tf.float32)
        predicted_text_indices = tf.keras.backend.cast(tf.keras.backend.argmax(predicted_text, axis=-1), tf.float32)

        # Вычисление манхэттенского расстояния между эмбеддингами текста
        true_embeddings = self.text_encoder(true_text_indices)
        predicted_embeddings = self.text_encoder(predicted_text_indices)

        embedding_dist = tf.keras.backend.sum(tf.keras.backend.abs(true_embeddings - predicted_embeddings), axis=-1)
        embedding_dist = self.reshape_loss(embedding_dist)

        # Возвращаем среднее значение ошибки по батчу
        return (cross_entropy_loss + embedding_dist_weight * embedding_dist) / 2

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            # Прямой проход
            true_text, predicted_text = self(inputs, training=True)
            loss = self.compute_loss(true_text, predicted_text)

        # Обратный проход
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Отслеживание ошибки
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, inputs):
        true_text, predicted_text = self(inputs, training=False)
        loss = self.compute_loss(true_text, predicted_text)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

