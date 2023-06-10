"""Importar librerías de trabajo"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DICT_SIZE = 25000

def tokenize_data(imdb_df):

    X = imdb_df['review']
    y = imdb_df['sentiment']

    '''Se crean los conjuntos de entrenamiento y de evaluación'''
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def get_max_length():
        '''Función para obtener la máxima longitud de las reseñas basada en el promedio de estas'''
        review_length = []
        for review in x_train:
            review_length.append(len(review))

        return int(np.ceil(np.mean(review_length)))

    '''Se crear un Tokenizer que servirá de diccionario para etiquetas las palabra que usará x_train como fuente'''
    token = Tokenizer(num_words = DICT_SIZE, lower=False,  oov_token='OOV') 
    token.fit_on_texts(x_train)
   

    '''Se convierten los texto a secuencia númerica y se aplica padding al conjunto de entrenamiento'''
    x_train = token.texts_to_sequences(x_train)
    x_test = token.texts_to_sequences(x_test)

    max_seq_length = get_max_length()

    x_train = pad_sequences(x_train, maxlen=max_seq_length, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=max_seq_length, padding='post', truncating='post')

    X_train = np.array(x_train).astype('int32')
    y_train = np.array(y_train).reshape((-1,1))
    X_test = np.array(x_test).astype('int32')
    y_test = np.array(y_test).reshape((-1,1))

    return (X_train, X_test, y_train, y_test, DICT_SIZE, max_seq_length)

