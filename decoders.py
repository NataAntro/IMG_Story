import slicer_layer
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input, Bidirectional, RepeatVector, Concatenate, Activation, Dot, TimeDistributed, Reshape
from tensorflow.keras.backend import variable
from tensorflow.keras.activations import softmax
from tensorflow.keras import Model

# Функция для применения softmax по последней оси
def apply_softmax(x):
    return softmax(x, axis=-1)

# Вспомогательная функция для выполнения одного шага механизма внимания
def attention_step(a, s_prev, repeator, concatenator, densor1, densor2, activator, dotor):
    """
    Выполняет один шаг механизма внимания: возвращает вектор контекста, вычисленный как скалярное произведение
    весов внимания "alphas" и скрытых состояний "a" Bi-LSTM.

    Аргументы:
    a -- скрытое состояние выхода Bi-LSTM, numpy-массив формы (m, in_seq_len, 2*pre_att_LSTM_size)
    s_prev -- предыдущее скрытое состояние (post-attention) LSTM, numpy-массив формы (m, post_att_LSTM_size)

    Возвращает:
    context -- вектор контекста, входящий в следующую ячейку (post-attention) LSTM
    """

    # Используем repeator для повторения s_prev до формы (m, num_words, post_att_LSTM_size)
    s_prev = repeator(s_prev)

    # Используем concatenator для объединения a и s_prev по последней оси
    concat = concatenator([a, s_prev])

    # Используем densor1 для пропускания concat через небольшую полносвязанную сеть для вычисления "промежуточных энергий" e
    e = densor1(concat)

    # Используем densor2 для пропускания e через небольшую полносвязанную сеть для вычисления "энергий"
    energies = densor2(e)

    # Используем "activator" для вычисления весов внимания "alphas"
    alphas = activator(energies)

    # Используем dotor вместе с "alphas" и "a" для вычисления вектора контекста для передачи в следующую ячейку post-attention LSTM
    context = dotor([alphas, a])

    return context

# Функция для создания модели декодера с механизмом внимания
def create_decoder_with_attention(
        input_seq_len, output_seq_len, num_input_features, pre_attention_lstm_size, post_attention_lstm_size, vocab_size
):
    cell_initial_state = [0] * post_attention_lstm_size
    initial_state_tensor = variable(cell_initial_state)
    x = Input(shape=(input_seq_len, num_input_features))
    s0 = Input(tensor=initial_state_tensor, name='s0')
    c0 = Input(tensor=initial_state_tensor, name='c0')

    # Инициализация слоев для механизма внимания
    repeator = RepeatVector(input_seq_len)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    densor2 = Dense(1, activation="relu")
    activator = Activation(apply_softmax, name='attention_weights')
    dotor = Dot(axes=1)

    s = s0
    c = c0

    # Инициализация пустого списка выходов
    outputs = []

    # Определение Bi-LSTM для предварительного внимания
    pre_attention_lstm = Bidirectional(LSTM(pre_attention_lstm_size, return_sequences=True))(x)

    # Определение LSTM для post-attention
    post_attention_lstm = LSTM(post_attention_lstm_size, return_state=True)

    # Softmax слой выхода
    softmax_output_layer = Dense(vocab_size, activation=apply_softmax)

    # Выполнение итераций по каждому шагу последовательности
    for t in range(output_seq_len):
        # Выполнение одного шага механизма внимания
        context = attention_step(pre_attention_lstm, s, repeator, concatenator, densor1, densor2, activator, dotor)

        # Применение ячейки post-attention LSTM к вектору контекста
        s, _, c = post_attention_lstm(context, initial_state=[s, c])

        # Применение Dense слоя к выходу post-attention LSTM
        out = softmax_output_layer(s)

        # Добавление "out" в список выходов
        outputs.append(out)

    # Создание экземпляра модели с тремя входами и возвратом списка выходов
    model = Model(inputs=[x, s0, c0], outputs=[outputs, s, c])

    return model

# Функция для создания модели декодера на основе BiLSTM
def create_bilstm_decoder(
        num_photos, num_words, num_input_features, lstm_units, vocab_size
):

    # Входной слой формы (None, num_photos, num_input_features)
    x_input = Input(shape=(num_photos, num_input_features))
    final_softmax_outputs = []

    # Не тренировать отдельные сети LSTM
    bi_lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))
    lstm_layer = LSTM(lstm_units, return_sequences=True, return_state=True)

    # Вспомогательные слои для создания последовательности истории для каждого фото
    reshape_layer = Reshape(target_shape=(1, num_words, vocab_size))
    concat_layer = Concatenate(axis=1)

    # Нет начального состояния для первого тензора
    previous_output = None
    previous_cell_state = None

    # Разбиение на список срезов формы (None, num_input_features)
    tensors_list = slicer_layer.TimestepSliceLayer()(x_input)

    for tensor in tensors_list:
        # Создание последовательности из num_words для каждого фото => тензоры формы (None, num_words, num_input_features)
        repeated_tensor = RepeatVector(num_words)(tensor)

        if previous_output is not None and previous_cell_state is not None:
            # Первым слоем Bi-LSTM с начальным состоянием из предыдущего шага
            initial_state = [previous_output, previous_cell_state, previous_output, previous_cell_state]
        else:
            # Нули в качестве начального состояния
            initial_state = None

        # Первый слой Bi-LSTM
        lstm_output = bi_lstm_layer(repeated_tensor, initial_state=initial_state)

        # Второй слой LSTM
        lstm_output_seq, previous_output, previous_cell_state = lstm_layer(lstm_output)

        # Softmax слой выхода => (None, num_words, vocab_size)
        softmax_output = TimeDistributed(Dense(vocab_size, activation='softmax'))(lstm_output_seq)
        reshaped_output = reshape_layer(softmax_output)
        final_softmax_outputs.append(reshaped_output)

    # (None, num_photos, num_words, vocab_size)
    final_output = concat_layer(final_softmax_outputs)

    # Создание экземпляра модели с входом x_input и выходом final_output
    model = Model(inputs=x_input, outputs=final_output)

    return model
