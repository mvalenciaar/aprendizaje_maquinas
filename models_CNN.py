from data_tokenizing import tokenize_data
from data_nlp import show_nlp_results
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import time
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import ParameterGrid
from keras.callbacks import EarlyStopping

(X_train, X_test, y_train, y_test, embedding_matrix, max_len) = tokenize_data()

X_train, X_val = X_train[:-5000], X_train[-5000:]
y_train, y_val = y_train[:-5000], y_train[-5000:]

embedding_dim = 100

def create_model(filters = 64, kernel_size = 3, strides=1, units = 256, 
                 optimizer='adam', rate = 0.25):
    model = Sequential()
    # Embedding layer
    model.add(Embedding(len(embedding_matrix) , 100, weights=[embedding_matrix], input_length= max_len))
    # Convolutional Layer(s)
    model.add(Conv1D(filters = filters, kernel_size = kernel_size, strides= strides, 
                     padding='valid', activation= 'relu'))
    model.add(MaxPooling1D(pool_size = 7))
    model.add(GlobalMaxPooling1D())
    # Dense layer(s)
    model.add(Dense(units = units, activation= 'relu'))
    model.add(Dropout(rate))
    # Output layer
    model.add(Dense(1, activation= 'sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer= optimizer,
                  metrics=['accuracy'])
    return model
# Build the model
model = KerasClassifier(build_fn= create_model)

# Set the hyperparameters
filters = [128] #[64, 128, 256]
kernel_size = [3] #[3, 5, 7]
strides= [1] # [1, 2, 5]
Dense_units = [32, 128, 512]
rate_dropouts = [0.25] #[0.1, 0.25, 0.5]
optimizers = ['adam'] #['adam','rmsprop']
epochs = [5]
batches = [64] #[32, 64, 128]
# ----------------------------------------------
# Exhaustive Grid Search
param_grid = dict(optimizer= optimizers, epochs= epochs, batch_size= batches,
                  filters = filters, kernel_size = kernel_size, strides = strides, 
                  units = Dense_units, rate = rate_dropouts)

grid = ParameterGrid(param_grid)
param_sets = list(grid)

param_scores = []
for params in grid:

    print(params)
    model.set_params(**params)

    earlystopper = EarlyStopping(monitor='val_accuracy', patience= 0, verbose=1)
    
    history = model.fit(X_train, y_train,
                        shuffle= True,
                        validation_data=(X_val, y_val),
                        callbacks= [earlystopper])

    param_score = history.history['val_accuracy']
    param_scores.append(param_score[-1])
    print('+-'*50) 

p = np.argmax(np.array(param_scores))
print('param_scores:', param_scores)
print("best score:", param_scores[p])
# Choose best parameters
best_params = param_sets[p]
print("best parameter set", best_params)

model.set_params(**best_params)
model.fit(X_train, y_train)

print("Test accuracy = %f%%" % (accuracy_score(y_test, model.predict(X_test))*100))