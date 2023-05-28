"""Importar librerías"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import ModelCheckpoint   # save model
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay, classification_report # for model evaluation metrics

# ARCHITECTURE
EMBED_DIM = 32
LSTM_OUT = 64

'''Función para guardar el modelo localmente si hay mejora'''
checkpoint = ModelCheckpoint(
    'models/LSTM.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

def run_lstm_model(X_train, X_test, y_train, y_test, total_words, max_seq_length):
  '''Construye, entrena y evalúa el modelo'''
  model_lstm = build_lstm_model(total_words, max_seq_length)
  model_lstm, history_lstm = train_model(model_lstm, X_train, y_train)
  plot_loss_curves(history_lstm)
  test_model(model_lstm, X_test, y_test)
  plot_confusion_matrix(model_lstm, X_test, y_test)

# Build the model
def build_lstm_model(total_words, max_seq_length):
    '''Crea modelo LSTM simple'''
    model_lstm = Sequential ([
            Embedding(total_words, EMBED_DIM, input_length = max_seq_length),
            LSTM(LSTM_OUT, dropout = 0.2, recurrent_dropout = 0.2),
            Dense(1, activation = 'sigmoid')
        ])

    # Set the training parameters

    # Compile the model
    model_lstm.compile(loss ='binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    return model_lstm

# Train the model
def train_model(model_lstm, X_train, y_train):
  '''Entrena el modelo de forma exhaustiva para encontrar los mejores parámetros'''
  history_lstm = model_lstm.fit(X_train, y_train, epochs = 10, validation_split = 0.2, callbacks=[checkpoint])
  return (model_lstm, history_lstm)

def test_model(model_lstm, X_test, y_test):
  '''Evalúa el modelo y devuelve la precisión y f1 score'''
  print('================================Testing set================================')
  _, accuracy = model_lstm.evaluate(X_test, y_test)
  print(f"Test accuracy: {accuracy}")

  y_pred = model_lstm.predict(X_test)
  # Convert the predicted probabilities to binary labels
  y_pred = (y_pred > 0.5).astype(int)
  # Compute the F1 score
  f1 = f1_score(y_test, y_pred)
  # Print the F1 score
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
  plt.title('Confusion Matrix for LSTM')
  plt.show()

  print(classification_report(y_test, predictions))



