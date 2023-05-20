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
    y = np.array(imdb_df['sentiment'])

    '''Se crean los conjuntos de entrenamiento y de evaluación'''
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    '''Se procede a realizar padding para que todas las reseñas tengan la misma longitud.
    Se definine los paramétros para el padding de las reseñas'''
    oov_tok = "<OOV>"
    trunc_type='post'
    padding_type='post'
    max_seq_length = 1000

    '''Se crear un Tokenizer que servirá de diccionario para etiquetas las palabra. 
    Tendrá longitud máxima de 5000 y usará el conjunto de entramiento como fuente'''
    tokenizer = Tokenizer(num_words = 5000, oov_token = oov_tok)
    tokenizer.fit_on_texts(x_train)

    '''Se convierten los texto a secuencia númerica y se aplica padding al conjunto de entrenamiento'''
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    x_train = pad_sequences(x_train, maxlen=max_seq_length, padding=padding_type, truncating=trunc_type)

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

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train, x_test, y_test, embedding_matrix, max_len)

