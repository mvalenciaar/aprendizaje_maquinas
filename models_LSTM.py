"""Importar librerías"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, LSTM,  Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import ParameterGrid, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # save model
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay, classification_report # for model evaluation metrics

# ARCHITECTURE
EMBED_DIM = 32
LSTM_OUT = 64

# Parámetros para GridSearch
param_grid = {
    'batch_size': [32, 64],
    'epochs': [5, 10],
    'dropout_rate': [0.25, 0.5, 0.75],
}

callback_list = [
                 ModelCheckpoint(
                     filepath = 'models/LSTM.h5',
                     monitor = 'val_accuracy',
                     verbose = 1,
                     save_best_only = True,
                     save_weights_only = False,
                     mode = 'max',
                     period = 1
                 ),
                 EarlyStopping(
                    monitor = 'val_accuracy',
                    patience = 5,
                    verbose = 1,
                    mode = 'max',
                    baseline = 0.5,
                    restore_best_weights = True
                 ),
]

def run_lstm_model(X_train, X_test, y_train, y_test, total_words, max_seq_length):
  '''Construye, entrena y evalúa el modelo'''
  model_lstm = KerasClassifier(build_fn = build_lstm_model, total_words = total_words, max_seq_length = max_seq_length)
  model_lstm, history_lstm = train_model(model_lstm, X_train, y_train)
  plot_loss_curves(history_lstm)
  test_model(model_lstm, X_test, y_test)
  plot_confusion_matrix(model_lstm, X_test, y_test)

def build_lstm_model(total_words, max_seq_length, dropout_rate = 0.75, units = 256):
    '''Crea modelo LSTM simple'''
    model_lstm = Sequential ([
            Embedding(total_words, EMBED_DIM, input_length = max_seq_length),
            Dropout(dropout_rate),
            LSTM(LSTM_OUT),
            Dense(units=units, activation='relu'),
            Dropout(dropout_rate),
            Dense(units=1, activation='sigmoid')
        ])

    model_lstm.summary()

    model_lstm.compile(loss ='binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    return model_lstm

def train_model(model_lstm, X_train, y_train):
  '''Código comentado. Descomentar para revisar proceso de búsqueda exhaustiva'''
  '''Entrena el modelo de forma exhaustiva para encontrar los mejores parámetros'''
  '''start = time.time()
  grid = GridSearchCV(estimator= model_lstm, param_grid=param_grid, n_jobs=-1, cv=3)
  grid_result = grid.fit(X_train,y_train)

  #Mostrar resultados finales
  print('time for grid search = {:.0f} sec'.format(time.time()-start))
  display_cv_results(grid_result)

  # Cargar el mejor modelo
  mlp = grid_result.best_estimator_

  # Entrenar el conjunto de entrenamiento con el mejor modelo obtenido
  history = mlp.fit(
      X_train,
      y_train,
      validation_split = 0.3,
      epochs = 10,
      callbacks = callback_list    
  )

  return mlp, history'''

  #Código usando hiperparámetros obtenidos en grid search
  history = model_lstm.fit(
      X_train,
      y_train,
      validation_split = 0.3,
      epochs = 7,
      batch_size = 64,
      callbacks = callback_list    
  )

  return model_lstm, history


def display_cv_results(search_results):
  '''Función para mostrar los resultados del GridSearch'''
  print('Mejor Puntuación = {:.4f} usando {}'.format(search_results.best_score_, search_results.best_params_))
  means = search_results.cv_results_['mean_test_score']
  stds = search_results.cv_results_['std_test_score']
  params = search_results.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print('Precisión promedia en validación +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))  

def test_model(model_lstm, X_test, y_test):
  '''Evalúa el modelo y devuelve la precisión y f1 score'''
  print('================================Testing set================================')
  _, accuracy = model_lstm.model.evaluate(X_test, y_test)
  print(f"Precisión en el conjunto de evaluación: {accuracy}")

  y_pred = model_lstm.predict(X_test)
  # Convertir etiquetas predichas en etiquetas binarias
  y_pred = (y_pred > 0.5).astype(int)
  f1 = f1_score(y_test, y_pred)
  print("F1 score:", f1)


def plot_loss_curves(model):
    
    '''
      Crea curvas de función de pérdida para las métricas de entrenamiento y evaluación
    '''

    train_loss = model.history['loss']
    val_loss = model.history['val_loss']

    train_accuracy= model.history['accuracy']
    val_accuracy= model.history['val_accuracy']

    epochs=range(1,len(model.history['loss'])+1)
    plt.figure(figsize=(20,5))

    # plot loss data
    plt.subplot(1,2,2)
    plt.plot(epochs,train_loss,label="training_loss")
    plt.plot(epochs,val_loss,label="validation_loss")
    plt.title("Loss curves",size=20)
    plt.xlabel('epochs',size=20)
    plt.ylabel('loss',size=20)
    plt.legend(fontsize=15);
    plt.show()
    # plt.show()

    
    # plot accuracy data
    plt.subplot(1,2,1)
    plt.plot(epochs,train_accuracy,label="training_acc")
    plt.plot(epochs,val_accuracy,label="validation_acc")
    plt.title("Accuracy curves",size=20)
    plt.xlabel('epochs',size=20)
    plt.ylabel('Accuracy',size=20)
    plt.tight_layout()
    plt.legend(fontsize=15);
    plt.show()
  
def plot_confusion_matrix(model, X_test, y_test):
  '''Grafica de matriz de confusión'''
  predictions = model.predict(X_test)
  predictions = (predictions > 0.5).astype(np.float32) 

  cfm = confusion_matrix(y_test, predictions)
  cm_display = ConfusionMatrixDisplay(confusion_matrix = cfm, display_labels = ['Negative', 'Positive'])
  cm_display.plot()
  plt.title('Matriz de Confusión para modelo LSTM')
  plt.show()

  print(classification_report(y_test, predictions))



