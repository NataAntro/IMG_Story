# Импортируем модуль cli из web под именем stcli
from streamlit.web import cli as stcli
# Импортируем модуль sys
import sys
# Из модуля streamlit импортируем модуль runtime
from streamlit import runtime
# Проверяем существование runtime
runtime.exists()

import requests
import streamlit as st
import os
import pickle
import tensorflow as tf

# Загружаем модель из бинарного файла VIST.bin
with open('VIST.bin', 'rb') as f:
    model = pickle.load(f)

def upload_and_process_image(image_data):
    client_id = 'API_KEY'
    headers = {'Authorization': f'Client-ID {client_id}'}
    
    url = 'https://api.imgur.com/3/image'
    files = {'image': image_data}
    response = requests.post(url, headers=headers, files=files)
    img_url = ''

    if response.status_code == 200:
        img_url = response.json()['data']['link']
        st.write('URL изображения:', img_url)
    else:
        st.error('Ошибка загрузки изображения: ' + str(response.status_code))
        return None
    return img_url

def preprocess_image(image_url):
    # Предобработка изображения
    response = requests.get(image_url)
    image = tf.image.decode_image(response.content, channels=3)
    image = tf.image.resize(image, [299, 299])
    image = tf.expand_dims(image, axis=0)  # Добавление размерности батча
    return image

def generate_narrative(model, image_urls):
    # Генерация нарратива с использованием модели
    preprocessed_images = [preprocess_image(url) for url in image_urls]
    batch_images = tf.concat(preprocessed_images, axis=0)
    narrative = model.predict(batch_images)
    return narrative

def translate_to_russian(text):
    # Функция для перевода текста на русский язык
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        "q": text,
        "source": "en",
        "target": "ru",
        "format": "text",
        "key": os.getenv('API_KEY')
    }
    response = requests.post(url, params=params)
    if response.status_code == 200:
        translated_text = response.json()["data"]["translations"][0]["translatedText"]
        return translated_text
    else:
        st.error('Ошибка перевода текста: ' + str(response.status_code))
        return text

def VIST(image_urls):
    # Вызов модели для генерации нарратива
    narrative = generate_narrative(model, image_urls)
    translated_narrative = translate_to_russian(narrative)
    return translated_narrative

def main():
    st.title("📸 Создание автоматизированного повествования на основе последовательности изображений")
    st.write("Пожалуйста, загрузите ровно 5 изображений для создания повествования.")
    uploaded_files = st.file_uploader("Загрузить изображения", type=["jpg", "png"], accept_multiple_files=True)
    if uploaded_files:
        if len(uploaded_files) != 5:
            st.error("Вы должны загрузить ровно 5 изображений.")
            return

        img_urls = []
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.getvalue()
            img_url = upload_and_process_image(bytes_data)
            if img_url:
                img_urls.append(img_url)
        
        if img_urls:
            narrative = VIST(img_urls)
            st.text_area("Повествование на основе ваших изображений:", narrative, height=200)
        else:
            st.error("Не удалось получить повествование.")

    st.write("Приложение выполнено в рамках магистерской диссертации Антроповой Н.Г.")

# Если скрипт запускается напрямую
if __name__ == '__main__':
    # Если runtime существует
    if runtime.exists():
        # Вызываем функцию main()
        main()
    # Если runtime не существует
    else:
        # Устанавливаем аргументы командной строки
        sys.argv = ["streamlit", "run", sys.argv[0]]
        # Выходим из программы с помощью функции main() из модуля stcli
        sys.exit(stcli.main())
