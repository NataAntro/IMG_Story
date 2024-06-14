import os
import shutil
import json
import tensorflow as tf
from PIL import Image

# Пути к JSON файлам для различных наборов данных
train_jsons_path = './data/train'
dev_jsons_path = './data/dev'
test_jsons_path = './data/test'

# Пути к директориям с изображениями для различных наборов данных
train_images_path = './data/train/images'
dev_images_path = './data/dev/images'
test_images_path = './data/test/images'

# Директория для хранения TFRecord файлов
tfrecords_path = './data/tfrecords'

# Префиксы для TFRecord файлов
train_files_prefix = os.path.join(tfrecords_path, "train")
valid_files_prefix = os.path.join(tfrecords_path, "valid")
test_files_prefix = os.path.join(tfrecords_path, "test")

# Основная директория данных VIST
VIST_base_path = 'C:\\Users\\Bacca\\Desktop\\NataAntro\\data\\VIST'

# Количество изображений в одной истории
num_images_per_story = 5

# URL и контрольные суммы архивов изображений
images_tars = ['https://drive.google.com/u/0/uc?export=download&confirm=O0FM&id=0ByQS_kT8kViSeEpDajIwOUFhaGc']
               # 'https://drive.google.com/u/0/uc?export=download&confirm=neHH&id=0ByQS_kT8kViSZnZPY1dmaHJzMHc',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=R0ZC&id=0ByQS_kT8kViSb0VjVDJ3am40VVE',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=Fa16&id=0ByQS_kT8kViSTmQtd1VfWWFyUHM',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=e5zi&id=0ByQS_kT8kViSQ1ozYmlITXlUaDQ',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=Schi&id=0ByQS_kT8kViSTVY1MnFGV0JiVkk',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=JxmS&id=0ByQS_kT8kViSYmhmbnp6d2I4a2M',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=ZjfB&id=0ByQS_kT8kViSZl9aNGVuX0llcEU',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=Y-WX&id=0ByQS_kT8kViSWXJ3R3hsZllsNVk',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=J58f&id=0ByQS_kT8kViSR2N4cFpweURhTjg',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=BqIJ&id=0ByQS_kT8kViScllKWnlaVU53Skk',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=WaDl&id=0ByQS_kT8kViSV2QxZW1rVXcxT1U',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=UWXn&id=0ByQS_kT8kViSNGNPTEFhdGxkMnM']
images_sha1_sums = ['5569371225f30f1045816441afc05f6b9c346ea4']
                    # '267bb083c7d5405d0bbba6de7ebd4a0b5e3f1a78',
                    # '8e42e7aefccda721502df355608717a6270f6abc',
                    # '39765a3ac8f8fb25cf0587d83caeac25b906920c',
                    # '60fbf7fb870bb098e141e8f31a44e2854064a342',
                    # '3f15aa70fc3f4dedd908cf65574366830b1f91fe',
                    # 'bb0447c7163374b02b9f62d5e74d856a92122e04',
                    # '4c191eca9507bdb62d3c73c24d990e09f0912b4d',
                    # '2df0997ceb138b25b033c6ad2728f85deda765a4',
                    # '90da393652408c34f7d1e12bdad692d1cab4d0dc',
                    # 'ea6ebdee6067f750c6494ff91f1d9a37e12736a2',
                    # '46ee3a3520653d4e128d8d36975c49d7cb9f2a04',
                    # 'a5f1b9380450cbb918d88f4c2fde055844dad2b2']
                    
# Функция для создания признака из байтового значения
def create_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Функция для создания записи из данных истории и пути к изображениям
def create_tfrecord_entry(story_data, img_dir):
    image_paths = [os.path.join(img_dir, img['id'] + '.jpg') for img in story_data]
    captions = [img['caption'] for img in story_data]

    features = {
        "caption_0": create_bytes_feature(captions[0].encode()),
        "caption_1": create_bytes_feature(captions[1].encode()),
        "caption_2": create_bytes_feature(captions[2].encode()),
        "caption_3": create_bytes_feature(captions[3].encode()),
        "caption_4": create_bytes_feature(captions[4].encode()),
        "raw_image_0": create_bytes_feature(tf.io.read_file(image_paths[0]).numpy()),
        "raw_image_1": create_bytes_feature(tf.io.read_file(image_paths[1]).numpy()),
        "raw_image_2": create_bytes_feature(tf.io.read_file(image_paths[2]).numpy()),
        "raw_image_3": create_bytes_feature(tf.io.read_file(image_paths[3]).numpy()),
        "raw_image_4": create_bytes_feature(tf.io.read_file(image_paths[4]).numpy()),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))

# Функция для записи данных в TFRecord файл
def write_tfrecord(stories, index, img_dir, dataset):
    output_dir = os.path.join(VIST_base_path, dataset + '\\tfrecords')
    os.makedirs(output_dir, exist_ok=True)
    tfrecord_filename = f"{dataset}-{index}.tfrecord"
    tfrecord_filepath = os.path.join(output_dir, tfrecord_filename)

    with tf.io.TFRecordWriter(tfrecord_filepath) as writer:
        for story in stories:
            if len(story) == num_images_per_story:
                tfrecord_entry = create_tfrecord_entry(story, img_dir)
                writer.write(tfrecord_entry.SerializeToString())

# Функция для чтения примера из TFRecord файла
def parse_tfrecord(example_proto):
    feature_description = {
        "caption_0": tf.io.FixedLenFeature([], tf.string),
        "caption_1": tf.io.FixedLenFeature([], tf.string),
        "caption_2": tf.io.FixedLenFeature([], tf.string),
        "caption_3": tf.io.FixedLenFeature([], tf.string),
        "caption_4": tf.io.FixedLenFeature([], tf.string),
        "raw_image_0": tf.io.FixedLenFeature([], tf.string),
        "raw_image_1": tf.io.FixedLenFeature([], tf.string),
        "raw_image_2": tf.io.FixedLenFeature([], tf.string),
        "raw_image_3": tf.io.FixedLenFeature([], tf.string),
        "raw_image_4": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)
    for i in range(num_images_per_story):
        image_raw = example.pop(f"raw_image_{i}")
        example[f"image_{i}"] = tf.image.resize(tf.image.decode_jpeg(image_raw, channels=3), size=(299, 299))
    return example

# Функция для разделения TFRecord файла на части
def split_tfrecord_file(num_shards, dataset='dev'):
    tfrecord_dir = os.path.join(VIST_base_path, dataset + '\\tfrecords')
    tfrecord_path = os.path.join(tfrecord_dir, 'dev-1.tfrecord')

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    for i in range(num_shards):
        shard_path = os.path.join(tfrecord_dir, f"shard-part-{i}.tfrecord")
        writer = tf.data.experimental.TFRecordWriter(shard_path)
        writer.write(dataset.shard(num_shards, i))

# Функция для получения датасета
def load_dataset(file_pattern, batch_size):
    dataset = (tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
               .map(parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .shuffle(buffer_size=batch_size * 10)
               .batch(batch_size)
               .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
    return dataset

# Функция для проверки, является ли директория пустой
def is_directory_empty(path):
    if os.path.exists(path) and not os.path.isfile(path):
        return not os.listdir(path)
    return True

# Функция для нормализации JSON файла истории
def normalize_story_json(dataset='train'):
    working_dir = os.path.join(VIST_base_path, dataset)
    annotation_file = os.path.join(working_dir, f"{dataset}.story-in-sequence.json")
    normalized_file = os.path.join(working_dir, f"NORMALIZED_{dataset}.story-in-sequence.json")

    if os.path.isfile(normalized_file):
        return

    with open(annotation_file, encoding='utf-8') as file:
        data = json.load(file)

    stories = {}
    for annotation_list in data['annotations']:
        for annotation in annotation_list:
            story_id = annotation['story_id']
            photo_info = {'id': annotation['photo_flickr_id'],
                          'order': annotation['worker_arranged_photo_order'],
                          'caption': annotation['text']}
            if story_id in stories:
                stories[story_id]['photos'].append(photo_info)
            else:
                stories[story_id] = {'story_id': story_id, 'photos': [photo_info]}

    with open(normalized_file, 'w', encoding='utf-8') as file:
        json.dump(list(stories.values()), file)

# Функция для проверки валидности изображения
def validate_image(image_path):
    if os.path.isfile(image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                img.getdata()[0]
            return img.format == 'JPEG'
        except Exception:
            return False
    return False

# Функция для обработки данных VIST
def process_vist_data(dataset='train'):
    working_dir = os.path.join(VIST_base_path, dataset)
    images_dir = os.path.join(working_dir, 'images')
    normalized_json = os.path.join(working_dir, f"NORMALIZED_{dataset}.story-in-sequence.json")

    normalize_story_json(dataset)

    with open(normalized_json, encoding='utf-8') as file:
        data = json.load(file)

    if dataset == 'dev':
        images_dir = os.path.join(images_dir, 'val')
    else:
        images_dir = os.path.join(images_dir, dataset)

    split_index = 1
    for tar_url in images_tars:
        if is_directory_empty(images_dir):
            tar_filename = f"{dataset}-split{split_index}.tar.gz"
            tar_filepath = os.path.join(os.path.abspath(working_dir), tar_filename)

            tf.keras.utils.get_file(
                fname=tar_filepath,
                origin=tar_url,
                file_hash=images_sha1_sums[split_index],
                archive_format='tar',
                extract=True,
                cache_dir=tar_filepath
            )
            os.remove(tar_filepath)

        batch = []
        num_stories = 0
        max_stories = 750
        for story in data:
            story_images = []
            for photo in story['photos']:
                img_path = os.path.join(images_dir, photo['id'] + '.jpg')
                if validate_image(img_path):
                    story_images.append({'id': photo['id'], 'caption': photo['caption']})
            batch.append(story_images)
            num_stories += 1
            if dataset == 'dev' and num_stories == max_stories:
                break

        write_tfrecord(batch, split_index, images_dir, dataset)
        split_index += 1

