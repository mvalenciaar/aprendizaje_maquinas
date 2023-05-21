"""Importar librerías de trabajo"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from data_cleaning import process_data

def tokenize_data():
    imdb_df = process_data()

    X = imdb_df['review']
    y = imdb_df['sentiment']

    '''Se crean los conjuntos de entrenamiento y de evaluación'''
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    '''Se procede a realizar padding para que todas las reseñas tengan la misma longitud.
    Se definine los paramétros para el padding de las reseñas'''

    '''Se crear un Tokenizer que servirá de diccionario para etiquetas las palabra que usará x_train como fuente'''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)

    '''Se convierten los texto a secuencia númerica y se aplica padding al conjunto de entrenamiento'''
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    reviews_lens = [len(text) for text in x_train]

    max_seq_length = reviews_lens[np.argmax(reviews_lens)]

    x_train = pad_sequences(x_train, maxlen = max_seq_length)
    x_test = pad_sequences(x_test, maxlen = max_seq_length)

    X_train = np.array(x_train).astype('int32')
    y_train = np.array(y_train)
    X_test = np.array(x_test).astype('int32')
    y_test = np.array(y_test)

    '''Se tokenizan todas las palabras usando Word2Vec que permite representar vectorialmente palabras y similitud entre estas'''

    sentences = [review.split() for review in X]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=8,sg=0)

    '''Se genera una matriz de Word Embedding(Incrustación de palabras.'''
    vectors = []
    for sentence in sentences:
        vector = []
        for word in sentence:
            if word in model.wv.key_to_index:
                vector.append(model.wv.get_vector(word))
        if len(vector) > 0:
            vectors.append(np.mean(vector, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    embedding_matrix = np.array(vectors)

    max_len = max(len(seq) for seq in x_train)

    return (X_train, X_test, y_train, y_test, embedding_matrix, max_len)

