import os
import data_prep
import decoders
import text_encoders
import vision_encoders

os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import custom_model
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from data_prep import get_dataset
from vocabulary_builder import retrieve_tokenizer
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Пути к директориям с TFRecord файлами
train_tfrecords_path = os.path.join(os.getcwd(), 'data\\VIST\\train\\tfrecords')
dev_tfrecords_path = os.path.join(os.getcwd(), 'data\\VIST\\dev\\tfrecords')

# Пути для сохранения весов и контрольных точек моделей
weights_directory = 'C:\\Users\\Bacca\\Desktop\\NataAntro\\weights'
checkpoints_path = os.path.join(weights_directory, 'checkpoints')

vision_model_save_path = os.path.join(weights_directory, 'vision_model')
vision_model_weights_path = os.path.join(weights_directory, 'vision_model_weights')

text_model_save_path = os.path.join(weights_directory, 'text_model')
text_model_weights_path = os.path.join(weights_directory, 'text_model_weights')

custom_model_checkpoint_path = os.path.join(checkpoints_path, 'ckpted_custom_model_weights')

decoder_model_save_path = os.path.join(weights_directory, 'decoder_model')
decoder_model_weights_path = os.path.join(weights_directory, 'decoder_model_weights')

# Параметры обучения
num_epochs = 30
batch_size = 2
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
def create_or_load_model():
    tokenizer = retrieve_tokenizer()
    word_index = tokenizer.word_index

    if (os.path.isdir(vision_model_save_path) and os.path.isdir(decoder_model_save_path)
            and os.path.isdir(text_model_save_path)):
        text_encoder = keras.models.load_model(text_model_save_path)
        vision_encoder = keras.models.load_model(vision_model_save_path)
        decoder_model = keras.models.load_model(decoder_model_save_path)
        print('Модель восстановлена!')
    else:
        print('Создание новой ненатренированной модели...')
        text_encoder = text_encoders.create_glove_lstm_encoder(caption_len, word_index, lstm_units)
        vision_encoder = vision_encoders.create_xception_encoder(images_per_story, lstm_units)
        decoder_model = decoders.create_bilstm_decoder(images_per_story, caption_len, lstm_units, lstm_units, len(word_index) + 1)

    model = build_model(text_encoder, vision_encoder, decoder_model, word_index)

    # Если веса существуют, загружаем их
    if os.path.exists(text_model_weights_path) and os.path.exists(vision_model_weights_path) and os.path.exists(decoder_model_weights_path):
        text_encoder.load_weights(text_model_weights_path)
        vision_encoder.load_weights(vision_model_weights_path)
        decoder_model.load_weights(decoder_model_weights_path)
        print('Веса моделей загружены!')

    return model

# Функция для предсказания на dev наборе данных
def predict_on_dev():
    model = create_or_load_model()
    dev_dataset = get_dataset(os.path.join(dev_tfrecords_path, "dev-*.tfrecord"), batch_size)
    batch = next(iter(dev_dataset))

    expected_text_indices, got_text_softmax = model.predict(batch)
    got_text_softmax = tf.argmax(got_text_softmax, axis=-1).numpy()
    tokenizer = retrieve_tokenizer()

    print('======== Предсказания для истории ========')
    for story in got_text_softmax:
        print('ПРЕДСКАЗАННЫЕ ПОДПИСИ ДЛЯ НОВОЙ ИСТОРИИ\n')
        caption = tokenizer.sequences_to_texts(story.tolist())
        print(caption)

    print('\n\n======== Правильные подписи для истории ========')
    for story in expected_text_indices:
        print('ПРАВИЛЬНЫЕ ПОДПИСИ ДЛЯ НОВОЙ ИСТОРИИ\n')
        caption = tokenizer.sequences_to_texts(story.tolist())
        print(caption)

# Функция для оценки модели на dev наборе данных
def evaluate_on_dev(model):
    dev_dataset = get_dataset(os.path.join(dev_tfrecords_path, "dev-*.tfrecord"), batch_size)
    loss, acc = model.evaluate(dev_dataset, verbose=2)
    print("Необученная модель, точность: {:5.2f}%".format(100 * acc))

# Функция для обучения модели
def train_model():
    model = create_or_load_model()
    train_dataset = get_dataset(os.path.join(train_tfrecords_path, "train-*.tfrecord"), batch_size)
    dev_dataset = get_dataset(os.path.join(dev_tfrecords_path, "dev-*.tfrecord"), batch_size)

    # Создание callback для изменения скорости обучения
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )

    # Создание callback для ранней остановки обучения
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Обучение модели
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=dev_dataset,
        callbacks=[reduce_lr, early_stopping]
    )

    # Сохранение модели
    model.text_encoder.save_weights(text_model_weights_path, save_format='tf')
    model.decoder.save_weights(decoder_model_weights_path, save_format='tf')
    model.vision_encoder.save_weights(vision_model_weights_path, save_format='tf')

    model.text_encoder.save(text_model_save_path)
    model.decoder.save(decoder_model_save_path)
    model.vision_encoder.save(vision_model_save_path)

    # Построение графика ошибки
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Ошибка")
    plt.xlabel("Эпоха")
    plt.legend(["обучение", "валидация"], loc="upper right")
    plt.show()

    print(model.summary())

# Очистка текущей сессии Keras и запуск обучения
tf.keras.backend.clear_session()
train_model()
# predict_on_dev()
