"""Importar librerías"""
from data_tokenizing import tokenize_data
import matplotlib.pyplot as plt
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay, classification_report  # for model evaluation metrics

# ARCHITECTURE
EMBED_DIM = 32
LSTM_OUT = 64

callback_list = [
                 ModelCheckpoint(
                     filepath = 'models/CNN.h5',
                     monitor = 'val_accuracy',
                     verbose = 1,
                     save_best_only = True,
                     save_weights_only = False,
                     mode = 'max',
                     period = 1
                 ),
                 
                 EarlyStopping(
                    monitor = 'val_accuracy',
                    patience = 2,
                    verbose = 1,
                    mode = 'max',
                    baseline = 0.5,
                    restore_best_weights = True
                 ),

                 ReduceLROnPlateau(
                     monitor = 'val_loss',
                     factor = 0.2,
                     patience = 2,
                     verbose = 1,
                     mode = 'min',
                     cooldown = 1,
                     min_lr = 0
                 )
]

def run_cnn_model(X_train, X_test, y_train, y_test, total_words, max_seq_length):
    '''Construye, entrena y evalúa el modelo'''
    model_cnn = build_cnn_model(total_words, max_seq_length)
    model_cnn, history_cnn = train_model(model_cnn, X_train, y_train)
    plot_loss_curves(history_cnn)
    test_model_cnn(model_cnn, X_test, y_test)
    plot_confusion_matrix(model_cnn, X_test, y_test)

def build_cnn_model(total_words = 10000, max_seq_length = 130, filters = 64, rate = 0.35):
    '''Crea la red convolucional de una dimensión con una capa de max pooling y dropout'''
    model = Sequential([
        Embedding(total_words, EMBED_DIM, input_length = max_seq_length),
        Conv1D(filters = filters, kernel_size = 5, strides = 1, 
                     padding='valid', activation= 'relu'),
        MaxPooling1D(pool_size = 7),
        GlobalMaxPooling1D(),
        Dense(128, activation= 'relu'),
        Dropout(rate),
        Dense(1, activation= 'sigmoid')
    ])

    model.summary()

    # Compile the model
    model.compile(loss ='binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    return model

def train_model(model_cnn, X_train, y_train):  
    history_cnn = model_cnn.fit(X_train,
                    y_train,
                    epochs = 10,
                    batch_size = 64,
                    verbose = 1,
                    callbacks = callback_list,
                    validation_split = 0.3,
                    shuffle = True)
    return (model_cnn, history_cnn)
    

def test_model_cnn(model_cnn, X_test, y_test):
    '''Evalúa el modelo y devuelve la precisión y f1 score'''
    _, accuracy = model_cnn.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

    y_pred = model_cnn.predict(X_test)
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

    train_accuracy = model.history['accuracy']
    val_accuracy = model.history['val_accuracy']

    epochs = range(1,len(model.history['loss'])+1)
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

    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
  '''Grafica de matriz de confusión'''
  predictions = model.predict(X_test)
  predictions = (predictions > 0.5).astype(np.float32) 

  cfm = confusion_matrix(y_test, predictions)
  cm_display = ConfusionMatrixDisplay(confusion_matrix = cfm, display_labels = ['Negative', 'Positive'])
  cm_display.plot()
  plt.title('Matriz de confusión para modelo CNN')
  plt.show()

  print(classification_report(y_test, predictions))