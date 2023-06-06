from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def run_tree(X_train, X_test, y_train, y_test):
    """Build train and evaluate the model"""
    #Train the model
    tree_clf = dec_tree(X_train, y_train)
    #Evaluate the model
    pred_test, acc_score_train, acc_score_test, f1 = test_tree(X_train, X_test, y_train, y_test)
    print(f'F1 Score: {f1}')
    print(f'Train accuracy score: {acc_score_train}')
    print(f'Test accuracy score: {acc_score_test}')
    #Execute plots
    plot_confusion_matrix(y_test, pred_test)
    auc_score = plots(y_test, pred_test)
    print(f'AUC: {auc_score}')

def dec_tree(X_train, y_train, save_model=True, model_path = 'models/dec_tree.pkl'):
    """This function create and train the model of decision tree"""
    #Hyperparameters to optimize
    param_grid = {
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [None, 5, 10],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 3]
    }
    #Create the model and optimize hyperparameters
    clf = DecisionTreeClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    #Best params and tree
    best_params = grid_search.best_params_
    best_tree = DecisionTreeClassifier(**best_params)

    #Train the final model
    best_tree.fit(X_train, y_train)

    #Save the model
    if save_model:
        joblib.dump(best_tree, model_path)

    return best_tree

def test_tree(X_train, X_test, y_train, y_test, model_path = 'models/dec_tree.pkl'):
    """Function to evaluate the model"""
    #Load the model
    tree_clf = joblib.load(model_path)
    #Predictions of the data train and test
    pred_train = tree_clf.predict(X_train)
    pred_test = tree_clf.predict(X_test)
    #Calculate the score
    acc_score_train = accuracy_score(y_train, pred_train)
    acc_score_test = accuracy_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)

    return pred_test, acc_score_train, acc_score_test, f1

def plot_confusion_matrix(y_test, pred_test):
    """Confusion matrix"""
    cm = confusion_matrix(y_test, pred_test)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Negative', 'Positive'])
    cm_display.plot()
    plt.title('Matriz de confusión para modelo Regresión Logística')
    plt.show()

def plots(y_test, pred_test):
    #Curve ROC and AUC calculation
    fpr, tpr, thresholds = roc_curve(y_test, pred_test)

    #Calculate AUC value
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
