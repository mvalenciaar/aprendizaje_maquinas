"""Importar librerías"""
from data_tokenizing import tokenize_data
import matplotlib.pyplot as plt
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import ParameterGrid, GridSearchCV
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay # for model evaluation metrics

embedding_dim = 100

# Parámetros para GridSearch
filters = [64, 128] #[64, 128, 256]
rate_dropouts = [0.1, 0.25] #[0.1, 0.25, 0.5]
batches = [32, 64] #[32, 64, 128]
epochs = [5, 10]

def run_cnn_model(X_train, X_test, y_train, y_test, embedding_matrix, max_len):
    '''Construye, entrena y evalúa el modelo'''
    model_cnn = KerasClassifier(build_fn=build_cnn_model, embedding_matrix=embedding_matrix, max_len=max_len)
    model_cnn = train_model(model_cnn, X_train, y_train)
    test_model_cnn(model_cnn, X_test, y_test)
    plot_loss_curves(model_cnn)
    plot_confusion_matrix(model_cnn, X_test, y_test)

    model_cnn.save("model_cnn.h5")



def build_cnn_model(embedding_matrix = 10000, max_len = 1000, filters = filters, rate = 0.25):
    '''Crea la red convolucional de una dimensión con una capa de max pooling y dropout'''
    model = Sequential([
        Embedding(len(embedding_matrix) , 100, weights = [embedding_matrix], input_length = max_len, trainable = False),
        Conv1D(filters = filters, kernel_size = 5, strides = 1, 
                     padding='valid', activation= 'relu'),
        MaxPooling1D(pool_size = 7),
        GlobalMaxPooling1D(),
        Dense(128, activation= 'relu'),
        Dropout(rate),
        Dense(1, activation= 'sigmoid')
    ])

    # Compile the model
    model.compile(loss ='binary_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])

    return model

def train_model(model_cnn, X_train, y_train):
    '''Entrena el modelo de forma exhaustiva para encontrar los mejores parámetros'''
    '''X_train, X_val = X_train[:-5000], X_train[-5000:]
    y_train, y_val = y_train[:-5000], y_train[-5000:]

    # ----------------------------------------------
    # Exhaustive Grid Search

    param_grid = dict(epochs= epochs, batch_size= batches,
                  filters = filters, kernel_size = kernel_size, strides = strides, 
                  units = Dense_units, rate = rate_dropouts)

    grid = ParameterGrid(param_grid)
    param_sets = list(grid)

    print(model_cnn)

    param_scores = []
    for params in grid:

        print(params)
        model_cnn.set_params(**params)

        earlystopper = EarlyStopping(monitor='val_accuracy', patience= 0, verbose=1)

        history = model_cnn.fit(X_train, y_train,
                            shuffle= True,
                            validation_data=(X_val, y_val),
                            callbacks= [earlystopper])

        param_score = history.history['val_accuracy']
        param_scores.append(param_score[-1]) 

    p = np.argmax(np.array(param_scores))

    # Choose best parameters
    best_params = param_sets[p]

    model_cnn.set_params(**best_params)
    model_cnn.fit(X_train, y_train)
    return model_cnn'''

    param_grid = dict(batch_size= batches,
                  filters = filters,  rate = rate_dropouts, epochs=epochs)
    
    clr_cnn =  GridSearchCV(estimator=model_cnn,
                     param_grid=param_grid,
                     scoring='accuracy',
                     n_jobs = -1,
                     cv = 3)

    clr_cnn.fit(X_train, y_train)

    print('best paramters: ', clr_cnn.best_params_)

    return clr_cnn


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

def plot_loss_curves(history):
    
    '''
      Crea curvas de función de pérdida para las métricas de entrenamiento y evaluación
    '''

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1,len(history.history['loss'])+1)
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