import numpy as np
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
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    return context

def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:k]
    return sequences

# Функция для создания модели декодера с механизмом внимания и Beam Search
def create_decoder_with_attention_and_beam_search(input_seq_len, output_seq_len, num_input_features, pre_attention_lstm_size, post_attention_lstm_size, vocab_size, beam_width):
    cell_initial_state = [0] * post_attention_lstm_size
    initial_state_tensor = variable(cell_initial_state)
    x = Input(shape=(input_seq_len, num_input_features))
    s0 = Input(tensor=initial_state_tensor, name='s0')
    c0 = Input(tensor=initial_state_tensor, name='c0')

    repeator = RepeatVector(input_seq_len)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    densor2 = Dense(1, activation="relu")
    activator = Activation(apply_softmax, name='attention_weights')
    dotor = Dot(axes=1)

    s = s0
    c = c0

    outputs = []

    pre_attention_lstm = Bidirectional(LSTM(pre_attention_lstm_size, return_sequences=True))(x)
    post_attention_lstm = LSTM(post_attention_lstm_size, return_state=True)
    softmax_output_layer = Dense(vocab_size, activation=apply_softmax)

    for t in range(output_seq_len):
        context = attention_step(pre_attention_lstm, s, repeator, concatenator, densor1, densor2, activator, dotor)
        s, _, c = post_attention_lstm(context, initial_state=[s, c])
        out = softmax_output_layer(s)
        outputs.append(out)

    # Применение Beam Search на последнем шаге генерации
    last_output = outputs[-1]
    output_sequences = beam_search_decoder(last_output, beam_width)
    model = Model(inputs=[x, s0, c0], outputs=[outputs, s, c, output_sequences])

    return model
