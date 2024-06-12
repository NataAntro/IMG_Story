import os
import json
import pickle
import data_prep

from tensorflow.keras.preprocessing.text import Tokenizer

# Путь для сохранения токенизатора
tokenizer_save_path = os.path.join(os.getcwd(), 'data\\VIST\\tokenizer.pickle')
max_num_words = 20000

# Функция для загрузки текстового корпуса
def load_corpus(mode='fetch'):
    # Нормализуем JSON файлы с данными, если это еще не сделано
    data_prep.normalize_story_json('train')
    data_prep.normalize_story_json('test')
    data_prep.normalize_story_json('dev')

    # Подготовка путей к JSON файлам. Корпус текста должен быть построен на основе train, dev и test
    base_dir = os.getcwd()
    train_json_path = os.path.join(base_dir, 'data\\VIST\\train\\NORMALIZED_train.story-in-sequence.json')
    test_json_path = os.path.join(base_dir, 'data\\VIST\\test\\NORMALIZED_test.story-in-sequence.json')
    dev_json_path = os.path.join(base_dir, 'data\\VIST\\dev\\NORMALIZED_dev.story-in-sequence.json')
    json_paths = [train_json_path, test_json_path, dev_json_path]

    corpus = None
    for json_path in json_paths:
        with open(json_path, encoding='utf-8') as file:
            data = json.load(file)

        if mode == 'dump':
            corpus_save_path = os.path.join(base_dir, 'data\\VIST\\text_corpus.txt')
            with open(corpus_save_path, 'w', encoding='utf-8') as corpus_file:
                # Сохраняем текстовый корпус с использованием буферизации ОС
                for story in data:
                    for photo in story['photos']:
                        print(f" {photo['caption']}", file=corpus_file)
        else:
            # Загружаем в память и возвращаем
            corpus = []
            for story in data:
                for photo in story['photos']:
                    corpus.append(photo['caption'])
    return corpus

# Функция для создания и сохранения токенизатора
def create_and_save_tokenizer():
    tokenizer = Tokenizer(
        num_words=max_num_words,
        filters='"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=' ',
        char_level=False,
        oov_token='<UNK>'
    )
    corpus = load_corpus()
    tokenizer.fit_on_texts(corpus)

    # Сохраняем токенизатор в файл
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer

# Функция для загрузки токенизатора
def load_tokenizer():
    if os.path.isfile(tokenizer_save_path):
        # Токенизатор уже существует. Загружаем и возвращаем его
        with open(tokenizer_save_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    # Токенизатор не найден. Создаем и сохраняем новый токенизатор
    return create_and_save_tokenizer()

# Функция для поиска самой длинной последовательности слов
def find_longest_sequence():
    # Нормализуем JSON файлы с данными, если это еще не сделано
    data_prep.normalize_story_json('train')
    data_prep.normalize_story_json('test')
    data_prep.normalize_story_json('dev')

    # Подготовка путей к JSON файлам. Корпус текста должен быть построен на основе train, dev и test
    base_dir = os.getcwd()
    train_json_path = os.path.join(base_dir, 'data\\VIST\\train\\NORMALIZED_train.story-in-sequence.json')
    test_json_path = os.path.join(base_dir, 'data\\VIST\\test\\NORMALIZED_test.story-in-sequence.json')
    dev_json_path = os.path.join(base_dir, 'data\\VIST\\dev\\NORMALIZED_dev.story-in-sequence.json')
    json_paths = [train_json_path, test_json_path, dev_json_path]

    max_len = 0
    second_max_len = 0
    third_max_len = 0
    longest_caption = ""
    longest_photo_id = 0
    threshold = 200
    below_threshold = 0
    above_threshold = 0

    for json_path in json_paths:
        with open(json_path, encoding='utf-8') as file:
            data = json.load(file)

        for story in data:
            for photo in story['photos']:
                caption_len = len(photo['caption'])

                if caption_len < threshold:
                    below_threshold += 1
                else:
                    print(photo['caption'])
                    above_threshold += 1

                if caption_len > max_len:
                    third_max_len = second_max_len
                    second_max_len = max_len
                    max_len = caption_len
                    longest_caption = photo['caption']
                    longest_photo_id = photo['id']

    print('Самая длинная последовательность: ' + str(max_len))
    print('Последовательность: ' + longest_caption)
    print('ID фото: ' + str(longest_photo_id))

    print('Вторая по длине последовательность: ' + str(second_max_len))
    print('Третья по длине последовательность: ' + str(third_max_len))

    print('Подписи короче ' + str(threshold) + ': ' + str(below_threshold))
    print('Подписи длиннее ' + str(threshold) + ': ' + str(above_threshold))
