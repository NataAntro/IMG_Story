Привет, это Натали Антро и мой новый проект IMG_Story! <img src="https://media.giphy.com/media/hvRJCLFzcasrR4ia7z/giphy.gif" width="28px" height="28px">
<p>
<a href="https://image-story-b36fb7a799d2.herokuapp.com/" target="_blank">
    <img src="https://img.shields.io/badge/Launch-App-brightgreen" alt="Launch App">
</a>

<p>
<i>Выполнено в рамках магистерской диссертации. </i>
<p>
	
## Разработка и оценка алгоритмов компьютерного зрения для автоматизированного повествования на основе последовательностей изображений
	
<p>
<img src='https://github.com/MarikIshtar007/MarikIshtar007/blob/master/images/matrix.gif' alt='Awesome Matrix Code' align='right'/>
<p>
	
## Набор данных:
<p>

    Для реализации поставленных задач использован набор данных VIST (Visual Storytelling), который включает более 80 тысяч уникальных фотографий, сгруппированных в более 20 тысяч 
    последовательностей. Эти данные были нормализованы и предварительно обработаны, чтобы улучшить обучение модели.

<p>

<a href="https://paperswithcode.com/dataset/vist" target="_blank">
    <img src="https://img.shields.io/badge/View%20Dataset-brightgreen" alt="View Dataset">
</a>

## Разработка модели:

### Основная архитектура

Работа основывается на использовании архитектуры Sequence-to-Sequence (Seq2Seq) с применением LSTM. Эта архитектура позволяет эффективно учитывать контекст предыдущих изображений и текстов при генерации повествований. Модель состоит из трех основных компонентов: энкодера текста, энкодера изображений и декодера.

<img src='https://github.com/NataAntro/IMG_Story/blob/main/Architecture.png' alt='Awesome Matrix Code' align='centre'/>

### Энкодер текста

- **Преобразование текстовых данных**: Текстовые данные преобразуются в числовые векторы с помощью модели GloVe, которая создает 300-мерные векторы слов.
- **Обработка последовательностей**: Эти векторы затем передаются через несколько слоев LSTM, которые помогают модели запоминать и обрабатывать последовательности слов, учитывая их контекст.

### Энкодер изображений

- **Извлечение признаков**: Изображения обрабатываются с помощью предварительно обученной модели Xception для извлечения важных признаков, таких как формы и текстуры.
- **Анализ последовательностей**: Извлеченные признаки передаются через несколько слоев LSTM, которые помогают анализировать последовательность изображений, учитывая их взаимосвязь.

### Декодер

- **Хранение контекста**: Объединенные признаки текста и изображений передаются в ячейку памяти, которая хранит контекст всей последовательности.
- **Генерация текста**: Декодер использует несколько слоев LSTM для генерации текста, предсказывая следующее слово в предложении на основе контекста.
- **Функция активации**: Каждый выходной слой LSTM подключен к функции активации Softmax, которая выбирает наиболее вероятное следующее слово.


[![Typing SVG](https://readme-typing-svg.herokuapp.com?color=%2336BCF7&lines=Seq+2+Seq)](https://git.io/typing-svg)
 
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

 ## Результаты

### Основные достижения

- **Улучшение контекстуальной связности**: Работа демонстрирует, что модель, основанная на Seq2Seq с использованием LSTM, превосходит модели с использованием GRU по метрикам METEOR и ROUGE. Это является значительным достижением, так как демонстрирует улучшение качества генерируемых текстов.
<img src='https://github.com/NataAntro/IMG_Story/blob/main/Example.png' alt='Awesome Matrix Code' align='centre'/>

- **Эффективность модели**: Предложенная модель показывает примерно равную производительность по сравнению с более сложными архитектурами, такими как Greedy и AREL, что подтверждает ее эффективность без избыточной сложности.
  
<img src='https://github.com/NataAntro/IMG_Story/blob/main/Metrics.png' alt='Awesome Matrix Code' align='centre'/>

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
