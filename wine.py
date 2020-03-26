import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score


df = pd.read_csv('./winequality-red.csv')

X = df[df.columns[:-1]].values
y = df[df.columns[-1]]
classes = y.unique()
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = svm.SVC()

clf.fit(X_train, y_train)

y_score = clf.predict(X_test)

precision = precision_score(y_test, y_score, average='micro')
recall = recall_score(y_test, y_score, average='micro')
f1 = f1_score(y_test, y_score, average='micro')

print(f'Precision: {precision} - Recall: {recall} - F1: {f1}')
