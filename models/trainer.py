import os
import data_prep
import decoders
import text_encoders
import vision_encoders
import custom_model
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from data_prep import get_dataset
from vocabulary_builder import retrieve_tokenizer
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Установка детерминированного поведения
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

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
decoder_model_save_path = os.path.join(weights_directory, 'decoder_model')
decoder_model_weights_path = os.path.join(weights_directory, 'decoder_model_weights')

# Параметры обучения
num_epochs = 30
batch_size = 2
caption_len = 25
images_per_story = 5
lstm_units = 1024

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

    if os.path.exists(text_model_weights_path) and os.path.exists(vision_model_weights_path) and os.path.exists(decoder_model_weights_path):
        text_encoder.load_weights(text_model_weights_path)
        vision_encoder.load_weights(vision_model_weights_path)
        decoder_model.load_weights(decoder_model_weights_path)
        print('Веса моделей загружены!')

    return model

def train_model():
    model = create_or_load_model()
    train_dataset = get_dataset(os.path.join(train_tfrecords_path, "train-*.tfrecord"), batch_size)
    dev_dataset = get_dataset(os.path.join(dev_tfrecords_path, "dev-*.tfrecord"), batch_size)

    # Callbacks для управления скоростью обучения и остановки при отсутствии прогресса
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=dev_dataset,
        callbacks=[reduce_lr, early_stopping]
    )

    # Сохранение модели
    model.save_weights(checkpoints_path + '/model_weights.h5')
    model.save(checkpoints_path + '/model_complete.h5')

    # Построение графика ошибки
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return model

# Очистка текущей сессии Keras и запуск обучения
tf.keras.backend.clear_session()
model = train_model()
