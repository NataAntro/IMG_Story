Привет, это Натали Антро и мой новый проект IMG_Story! <img src="https://media.giphy.com/media/hvRJCLFzcasrR4ia7z/giphy.gif" width="28px" height="28px">
<p>
[![App Button](https://img.shields.io/badge/Launch-App-brightgreen)](https://image-story-b36fb7a799d2.herokuapp.com/)
<p>
<i>Выполнено в рамках магистерской диссертации. </i>
<p>
<b>Автоматизированное создание текстовых повествований на основе визуальных данных,</b> которое может анализировать последовательности изображений и генерировать связные текстовые описания.
<p>
<img src='https://github.com/MarikIshtar007/MarikIshtar007/blob/master/images/matrix.gif' alt='Awesome Matrix Code' align='right'/>
<p>
Как это работает?
<p>

    Пользователь загружает последовательность изображений через интерфейс приложения.<br>

    Изображения предобрабатываются: изменяется их размер, изображения нормализуются, и добавляется дополнительная размерность для обработки в батче.<br>

    Предобработанные изображения подаются на вход модели Xception для извлечения признаков.<br>

    Извлеченные признаки подаются на вход рекуррентной нейронной сети LSTM, которая учитывает контекст предыдущих изображений и текстов.<br>

    Модель генерирует текстовые повествования на основе визуальных данных.<br>

<p>
Технологии:
<p>

    Предварительно обученная сверточная нейронная сеть <a href="https://keras.io/api/applications/xception/" target="_blank">Xception</a> для извлечения признаков из изображений.<br>

    Рекуррентная нейронная сеть LSTM для генерации текстовых повествований.<br>

    Библиотека TensorFlow для работы с моделями глубокого обучения.<br>

    Библиотека NumPy для обработки данных.<br>

<p>

[![Typing SVG](https://readme-typing-svg.herokuapp.com?color=%2336BCF7&lines=Seq+2+Seq)](https://git.io/typing-svg)
 
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

```python
 class WhoAmI:
    user = 'NataAntro'
    current_adventure = ('URFU Student', 'AI/ML Engineer', 'QA Specialist (BaccaSoft)', 
			 'Metaverse Researcher', 'Digital artist')
    passions = [
        'Music',
        'Snowboarding',
        'Reading Pelevins books',
        'Gathering many good ideas into one great idea'
    ]

    @independent_method
    def home_base():
        return 'Moscow_Russia'

    @independent_method
    def on_agenda():
        return ['LearnAIandML', 'CreateStartUp', 'LiveinHappiness']
        # Assume 10 more awesome ambitions here  ;)

	
 ```
