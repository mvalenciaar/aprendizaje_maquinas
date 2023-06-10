#Importar librerias necesarias
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt

def run_lr_model(X_train, X_test, y_train, y_test, model_path = 'models/logreg_model.pkl'):
    '''Construye, entrena y evalúa el modelo'''
    #Crear y Entrenar el modelo
    model_log_reg = log_reg(X_train, y_train, model_path)
    #Evaluar el modelo
    pred_test, acc_score_train, acc_score_test, f1 = test_log_reg(X_train, X_test, y_train, y_test, model_path)
    print(f'F1 Score: {f1}')
    print(f'Train accuracy score: {acc_score_train}')
    print(f'Test accuracy score: {acc_score_test}')
    #Mostrar resultados gráficos
    plot_confusion_matrix(y_test, pred_test)
    auc_score = plots(y_test, pred_test)
    print(f'AUC: {auc_score}')


def log_reg(X_train, y_train, model_path, save_model=True):
    '''Esta función crea y entrena el modelo con hiperparámetros definidos'''
    log_reg = LogisticRegression(C=10.0, penalty='l2', max_iter=500)
    #Entrenar el modelo
    log_reg.fit(X_train, y_train)
    #Guardar el modelo
    if save_model:
            joblib.dump(log_reg, model_path)

    return log_reg

def test_log_reg(X_train, X_test, y_train, y_test, model_path):
    '''Esta función evalúa el modelo'''
    #Load the model
    log_reg = joblib.load(model_path)
    #Predecir en el conjunto de datos
    pred_train = log_reg.predict(X_train)
    pred_test = log_reg.predict(X_test)
    #Calcular la precisión en entrenamiento y evaluación
    acc_score_train = accuracy_score(y_train, pred_train)
    acc_score_test = accuracy_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)

    return pred_test, acc_score_train, acc_score_test, f1

def plot_confusion_matrix(y_test, pred_test):
    """Función para graficar la matrix de confusión"""
    cm = confusion_matrix(y_test, pred_test)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Negative', 'Positive'])
    cm_display.plot()
    plt.title('Matriz de confusión para modelo Regresión Logística')
    plt.show()

def plots(y_test, pred_test):
    '''Función para graficar la curva AUC y la curva ROC'''
    
    fpr, tpr, thresholds = roc_curve(y_test, pred_test)
    auc_score = auc(fpr, tpr)

    #Plot ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal de referencia
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.show()

    return auc_score



