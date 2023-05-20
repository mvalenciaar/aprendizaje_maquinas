"""Importar librerías"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings
from numpy import asarray
from numpy import zeros
import seaborn as sns
#from IPython.display import HTML,display
import random
from sklearn.model_selection import train_test_split
from data_cleaning import process_data
from data_tokenizing import tokenize_data
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Conv1D, MaxPool1D,GlobalMaxPooling1D, Dense, Dropout , LSTM,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import legacy

data_clean = process_data()
tok = tokenize_data()


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

# Train convert to numpy array

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Train the model
from sklearn.metrics import f1_score
with tf.device('/GPU:0'):
    early_stopping = EarlyStopping(patience=3)
    history_lstm = model_lstm.fit(x_train,y_train, epochs=20,verbose = 2,validation_split=0.1,callbacks=[early_stopping])


x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_test.shape)
print(x_test.dtype)

_, accuracy = model_lstm.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")

y_pred = model_lstm.predict(x_test)
# Convert the predicted probabilities to binary labels
y_pred = (y_pred > 0.5).astype(int)
# Compute the F1 score
f1 = f1_score(validation_labels, y_pred)
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


plot_loss_curves(history_lstm)



#Test model

def review_test(index , test_df):
    

    text = test_df['review'][index]
    display(HTML(f"<h5><b style='color:red'>Text: </b>{text}</h5>"))


    true_label = test_df['sentiment'][index]
    true_val = "negative" if true_label == 0 else "positive"
    display(HTML(f"<h5><b style='color:red'>Actual: </b>{true_val}</h5>"))

  #vectorizing the text by the pre-fitted tokenizer instance
    text = tokenizer.texts_to_sequences(text)

  #padding the text to have exactly the same shape as `embedding` input
    text = pad_sequences(text, maxlen=max_length, dtype='int32', value=0)


    sentiment = model_lstm.predict(text,batch_size=1,verbose = 2)[0]
    pred_val = "negative" if sentiment == 0 else "positive"
    display(HTML(f"<h5><b style='color:red'>Predicted: </b>{pred_val}</h5>"))

review_test(random.randint(1, 10000),imdb)

model_lstm.save("model_1.h5")

