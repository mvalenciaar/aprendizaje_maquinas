from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def run_tree(X_train, X_test, y_train, y_test):
    """Función para crear, entrenar y evaluar el modelo"""
    #Crear y Entrenar el modelo
    tree_clf = dec_tree(X_train, y_train)
    #Evaluar el modelo
    pred_test, acc_score_train, acc_score_test, f1 = test_tree(X_train, X_test, y_train, y_test)
    print(f'F1 Score: {f1}')
    print(f'Precisión en entrenamiento: {acc_score_train}')
    print(f'Precisión en evaluacuón: {acc_score_test}')
    #Mostrar resultados gráficos
    plot_confusion_matrix(y_test, pred_test)
    auc_score = plots(y_test, pred_test)
    print(f'AUC: {auc_score}')

def dec_tree(X_train, y_train, save_model=True, model_path = 'models/dec_tree.pkl'):
    """Esta función crea y entrena el modelo de árboles de decisión"""
    #Hiperparámetros para optimizar
    param_grid = {
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [None, 5, 10],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 3]
    }
    #Búsqueda exhaustiva de parámetros óptimos con GridSearchCV
    clf = DecisionTreeClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    #Eleccuón de mejores parámetros
    best_params = grid_search.best_params_
    best_tree = DecisionTreeClassifier(**best_params)

    #Entrenar el modelo
    best_tree.fit(X_train, y_train)

    #Guardar el modelo
    if save_model:
        joblib.dump(best_tree, model_path)

    return best_tree

def test_tree(X_train, X_test, y_train, y_test, model_path = 'models/dec_tree.pkl'):
    """Función para evaluar el modelo entrenado"""
    #Cargar el modelo
    tree_clf = joblib.load(model_path)
    #Predecir en el conjunto de datos
    pred_train = tree_clf.predict(X_train)
    pred_test = tree_clf.predict(X_test)
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
