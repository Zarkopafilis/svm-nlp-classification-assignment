import nltk

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras import regularizers

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import recall_score, f1_score, precision_score


print('Checking required nltk modules')
nltk.download('punkt')
nltk.download('stopwords')


df = pd.read_csv('onion-or-not.csv')

sentences = df['text'].values
labels = df['label'].values.astype('float')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

tokens = []

print('Tokenizing, Stemming, Removing Stopwords')
for i, sentence in enumerate(sentences):
    tokens = word_tokenize(sentence)
    stemmed = [stemmer.stem(x) for x in tokens]
    t = [w for w in stemmed if w not in stop_words]

    sentences[i] = ' '.join(t)
    tokens.append(t)

print('Calculating Tfidf')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences).astype('float')

train_test_limit = int(0.75 * X.shape[0])
X_test = X[train_test_limit:]
X = X[:train_test_limit]

y_test = labels[train_test_limit:]
y = labels[:train_test_limit]


def get_model():
    model = Sequential()
    model.add(Dense(256, input_dim=X.shape[1], activation='relu', kernel_regularizer=regularizers.l1(0.1),
                activity_regularizer=regularizers.l1(0.1)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.1),
                activity_regularizer=regularizers.l1(0.1)))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

print('Making Model')
clf = KerasClassifier(build_fn=get_model)

print('Training Model')
history = clf.fit(X.toarray(), y, epochs=30, validation_split=0.2)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print('Test Set Evaluation')
y_score = clf.predict(X_test.toarray())

precision = precision_score(y_test, y_score, average='micro')
recall = recall_score(y_test, y_score, average='micro')
f1 = f1_score(y_test, y_score, average='micro')

print(f'Precision: {precision} - Recall: {recall} - F1: {f1}')
