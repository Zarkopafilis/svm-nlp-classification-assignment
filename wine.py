import pandas as pd

from sklearn import svm
from sklearn.model_selection import GridSearchCV

def get_best_params():
    print("Loading best params")
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}

    df = pd.read_csv('./winequality-red.csv')

    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values

    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, verbose=1, n_jobs=-1)
    clf.fit(X, y)

    print('Grid Search Results')
    print(clf.cv_results_)

    print(f'Best params: {clf.best_params_}')
    return clf.best_params_
