import os
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from data_prep import get_dataset
from vocabulary_builder import retrieve_tokenizer
import decoders
import text_encoders
import vision_encoders
import custom_model

# Пути для сохранения моделей и бинарного файла
weights_directory = 'C:\\Users\\Bacca\\Desktop\\NataAntro\\weights'
binary_file_path = os.path.join(weights_directory, 'VIST.bin')

# Параметры обучения
caption_len = 25
images_per_story = 5
lstm_units = 1024

# Функция для создания модели
def build_model(text_encoder, vision_encoder, decoder, word_index):
    my_model = custom_model.CustomModel(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        decoder=decoder,
        words_to_idx=word_index,
        words_per_caption=caption_len,
    )
    my_model.compile(optimizer=keras.optimizers.Adam())
    return my_model

# Функция для создания новой или восстановления существующей модели
def create_model():
    tokenizer = retrieve_tokenizer()
    word_index = tokenizer.word_index

    text_encoder = text_encoders.create_glove_lstm_encoder(caption_len, word_index, lstm_units)
    vision_encoder = vision_encoders.create_xception_encoder(images_per_story, lstm_units)
    decoder_model = decoders.create_bilstm_decoder(images_per_story, caption_len, lstm_units, lstm_units, len(word_index) + 1)

    model = build_model(text_encoder, vision_encoder, decoder_model, word_index)
    return model

# Функция для сохранения модели в бинарный файл
def save_model_to_bin(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

# Функция для загрузки модели из бинарного файла
def load_model_from_bin(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Создание модели
model = create_model()

# Обучение модели
train_tfrecords_path = os.path.join(os.getcwd(), 'data\\VIST\\train\\tfrecords')
train_dataset = get_dataset(os.path.join(train_tfrecords_path, "train-*.tfrecord"), batch_size)
model.fit(train_dataset, epochs=20)

# Сохранение модели в бинарный файл
save_model_to_bin(model, binary_file_path)

# Загрузка модели из бинарного файла для проверки
loaded_model = load_model_from_bin(binary_file_path)

# Проверка загруженной модели
print(loaded_model.summary())
