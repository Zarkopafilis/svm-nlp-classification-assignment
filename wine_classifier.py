import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score

def run_experiment(df, params):
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]]
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = svm.SVC(**params)

    clf.fit(X_train, y_train)

    y_score = clf.predict(X_test)

    precision = precision_score(y_test, y_score, average='micro')
    recall = recall_score(y_test, y_score, average='micro')
    f1 = f1_score(y_test, y_score, average='micro')

    return precision, recall, f1
