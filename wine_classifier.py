import pandas as pd

from sklearn import svm
from sklearn.metrics import recall_score, f1_score, precision_score

def run_experiment(df, params, X_test, y_test):
    X_train = df[df.columns[:-1]].values
    y_train = df[df.columns[-1]].values

    # print(f'X_train = {X_train.shape} - y_train = {y_train.shape} - X_test = {X_test.shape} - y_test = {y_test.shape}')
    clf = svm.SVC(**params)

    clf.fit(X_train, y_train)

    y_score = clf.predict(X_test)

    precision = precision_score(y_test, y_score, average='micro')
    recall = recall_score(y_test, y_score, average='micro')
    f1 = f1_score(y_test, y_score, average='micro')

    return precision, recall, f1
