from data_cleaning import process_data
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout

data_clean = process_data()


# Build the model
with tf.device('/GPU:0'):
    model_lstm = Sequential ([
            Embedding(len(embedding_matrix),100, weights = [embedding_matrix], input_length = max_len,trainable=False),
            Bidirectional(GRU(100, return_sequences = True, dropout = 0.35, recurrent_dropout = 0.35)),
            Bidirectional(LSTM(100, return_sequences = True, dropout = 0.35, recurrent_dropout = 0.35)),
            Conv1D(100, 5, activation = 'relu'),
            GlobalMaxPooling1D(),
            Dense(100, activation = 'relu'),
            Dropout(0.5),
            Dense(1, activation = 'sigmoid')
        ])

    # Set the training parameters
    optimizer = Adam(learning_rate=0.001)
    model_lstm.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

model_lstm.summary()
tf.keras.utils.plot_model(model_lstm,show_shapes=True)

