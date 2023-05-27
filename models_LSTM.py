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
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay # for model evaluation metrics

def run_lstm_model(X_train, X_test, y_train, y_test, embedding_matrix, max_len):
  model_lstm = KerasClassifier(build_fn=build_lstm_model, embedding_matrix=embedding_matrix, max_len=max_len)
  model_lstm, history_lstm = train_model(model_lstm, X_train, y_train)
  test_model(model_lstm, X_test, y_test)
  plot_loss_curves(history_lstm)
  plot_confusion_matrix(model_lstm, X_test, y_test)

  model_lstm.save("model_lstm.h5")

# Build the model
def build_lstm_model(embedding_matrix, max_len):
    '''Crea modelo LSTM simple'''
    model_lstm = Sequential ([
            Embedding(len(embedding_matrix), 100, weights = [embedding_matrix], input_length = max_len,trainable=False),
            LSTM(100, dropout = 0.2, recurrent_dropout = 0.2),
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
  early_stopping = EarlyStopping(patience = 1)
  history_lstm = model_lstm.fit(X_train, y_train, epochs = 5, validation_split = 0.2, callbacks=[early_stopping])
  return (model_lstm, history_lstm)

def test_model(model_lstm, X_test, y_test):
  '''Evalúa el modelo y devuelve la precisión y f1 score'''
  _, accuracy = model_lstm.evaluate(X_test, y_test)
  print(f"Test accuracy: {accuracy}")

  y_pred = model_lstm.predict(X_test)
  # Convert the predicted probabilities to binary labels
  y_pred = (y_pred > 0.5).astype(int)
  # Compute the F1 score
  f1 = f1_score(y_test, y_pred)
  # Print the F1 score
  print("F1 score:", f1)


def plot_loss_curves(history):
    
    '''
      returns seperate loss curves for training and validation metrics
    '''

    train_loss=history.history['loss']
    val_loss=history.history['val_loss']

    train_accuracy=history.history['accuracy']
    val_accuracy=history.history['val_accuracy']

    epochs=range(1,len(history.history['loss'])+1)
    plt.figure(figsize=(20,5))

    # plot loss data
    plt.subplot(1,2,2)
    plt.plot(epochs,train_loss,label="training_loss")
    plt.plot(epochs,val_loss,label="validation_loss")
    plt.title("Loss curves",size=20)
    plt.xlabel('epochs',size=20)
    plt.ylabel('loss',size=20)
    plt.legend(fontsize=15);
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

    plt.title('Model Performance Curves')
  
def plot_confusion_matrix(model, X_test, y_test):
  '''Grafica de matriz de confusión'''
  predictions = model.predict(X_test)
  predictions = (predictions > 0.5).astype(np.float32) 

  cfm = confusion_matrix(y_test, predictions)
  cm_display = ConfusionMatrixDisplay(confusion_matrix = cfm, display_labels = ['Negative', 'Positive'])
  cm_display.plot()
  plt.show()



