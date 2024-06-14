import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Reshape, LSTM, TimeDistributed

# Количество изображений в одной истории
num_photos_per_story = 5

# Функция для получения слоя эмбеддингов GloVe
def create_embedding_layer(sentence_length, word_index):
    embeddings_dict = {}
    embedding_dim = 300

    # Путь к GloVe
    glove_file_path = 'C:\\Users\\Bacca\\Desktop\\NataAntro\\data\\VIST\\glove.6B.' + str(embedding_dim) + 'd.txt'
    with open(glove_file_path, encoding='utf8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                mask_zero=True,
                                input_length=sentence_length,
                                trainable=False)
    return embedding_layer

# Функция для создания текстового энкодера на основе BERT
def create_bert_text_encoder():
    # Загрузка модуля предобработки BERT
    preprocessing_layer = hub.KerasLayer(
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        name='bert_preprocessing'
    )

    # Загрузка предобученной модели BERT для использования в качестве энкодера
    bert_layer = hub.KerasLayer(
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2',
        trainable=False,
        name='bert'
    )

    # Запрет обучения энкодера BERT
    bert_layer.trainable = False

    # Входной слой для текстовых данных
    text_input = Input(shape=(), dtype=tf.string, name="text_input")

    # Предобработка текстовых данных
    preprocessed_text = preprocessing_layer(text_input)

    # Генерация эмбеддингов для предобработанного текста с использованием BERT
    bert_embeddings = bert_layer(preprocessed_text)["pooled_output"]

    # Создание модели энкодера текста
    return Model(inputs=text_input, outputs=bert_embeddings, name="bert_text_encoder")

# Функция для расчета манхэттенского расстояния
def calculate_manhattan_distance(A, B):
    return tf.keras.backend.sum(tf.keras.backend.abs(A - B), axis=1, keepdims=True)

# Функция для создания текстового энкодера на основе GloVe и LSTM
def create_glove_lstm_encoder(words_per_caption, word_index, lstm_units):
    # Входной слой для текстовых данных формы (None, num_photos_per_story, words_per_caption)
    text_input = Input(shape=(num_photos_per_story, words_per_caption), name="text_input")

    # Использование GloVe эмбеддингов => (None, num_photos_per_story, words_per_caption, 300)
    embedding_layer = create_embedding_layer(words_per_caption, word_index)
    glove_embeddings = TimeDistributed(embedding_layer)(text_input)

    # Изменение формы до (None, words_per_caption * num_photos_per_story, 300)
    reshaped_embeddings = Reshape((words_per_caption * num_photos_per_story, 300), input_shape=(num_photos_per_story, words_per_caption, 300))(glove_embeddings)

    # LSTM слой возвращает форму (None, words_per_caption * num_photos_per_story, lstm_units)
    lstm_output = LSTM(lstm_units, return_sequences=True)(reshaped_embeddings)

    # Создание модели энкодера текста
    return Model(inputs=text_input, outputs=lstm_output, name="glove_lstm_text_encoder")
