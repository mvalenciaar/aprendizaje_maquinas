#Importar librerias necesarias
from data_tokenizing import tokenize_data
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt

#(X_train, X_test, y_train, y_test, embedding_matrix, max_len) = tokenize_data() #Esta no va

def run_log_reg(X_train, X_test, y_train, y_test):
    """Build, train and evaluate the model"""
    #Train the model
    model_log_reg = log_reg(X_train, y_train)
    #Evaluate the model
    pred_test, acc_score_train, acc_score_test, f1 = test_log_reg(X_train, X_test, y_train, y_test)
    print(f'F1 Score: {f1}')
    print(f'Train accuracy score: {acc_score_train}')
    print(f'Test accuracy score: {acc_score_test}')
    #Execute plots
    auc_score = plots(y_test, pred_test)
    print(f'AUC: {auc_score}')


def log_reg(X_train, y_train, save_model=True, model_path = 'logreg_model.pkl'):
    """This function create and train the model, the hyperparameters where optimized before"""
    log_reg = LogisticRegression(C=10.0, penalty='l2', max_iter=500)
    #Train the model
    log_reg.fit(X_train, y_train)
    #Save the model
    if save_model:
        joblib.dump(log_reg, model_path)

    return log_reg

def test_log_reg(X_train, X_test, y_train, y_test, model_path = 'logreg_model.pkl'):
    """This function evaluate the model"""
    #Load the model
    log_reg = joblib.load(model_path)
    #Predictions of the data train and test
    pred_train = log_reg.predict(X_train)
    pred_test = log_reg.predict(X_test)
    #Calculating the accuracy
    acc_score_train = accuracy_score(y_train, pred_train)
    acc_score_test = accuracy_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)

    return pred_test, acc_score_train, acc_score_test, f1

def plots(y_test, pred_test):
    #Confusion Matrix
    cm = confusion_matrix(y_test, pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    #Curve ROC and AUC calc
    #Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, pred_test)

    #Calculate AUC value
    auc_score = auc(fpr, tpr)

    #Plot ROC curve
    # Graficar la curva ROC
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal de referencia
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()

    return auc_score



