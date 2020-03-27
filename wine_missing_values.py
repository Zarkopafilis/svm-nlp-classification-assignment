import random
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from wine_classifier import run_experiment
from sklearn.model_selection import train_test_split

from wine import get_best_params

pd.options.mode.chained_assignment = None  # default='warn'

# best_params = get_best_params() # takes some 10 min to run on my macbook
best_params = {'C': 1, 'gamma': 0.001, 'kernel': 'linear'}  # result of get_best_params()

df = pd.read_csv('./winequality-red.csv')

test_df = df.sample(frac=0.25, random_state=42)

X_test = test_df[test_df.columns[:-1]].values

test_df_no_ph = test_df.drop('pH', axis=1)
X_test_no_ph = test_df_no_ph[test_df_no_ph.columns[:-1]].values

y_test = test_df[test_df.columns[-1]].values

df = df.drop(test_df.index)

o_df = df.copy()
experiments = {}

experiments['all_data'] = o_df

no_ph = df.copy()
no_ph = no_ph.drop(['pH'], axis=1)

experiments['no_ph_col'] = no_ph

# destroy 33% of pH column
print("Destroying 33% of pH column")
destroy_idx = random.sample(df.index.tolist(), len(df) // 3)
bad_ph = df.index.isin(destroy_idx)
df['pH'].loc[destroy_idx] = 0

infer_from = df[~bad_ph]
no_ph_rows = df.iloc[bad_ph]

print('Filling empties with mean')
# fill empties with mean
mean_ph = infer_from['pH'].mean()
mean_df = o_df.copy()
mean_df['pH'].loc[destroy_idx] = mean_ph

experiments['mean_ph_impute'] = mean_df

# fill empties with regression
print('Filling empties with regression')
infer_from = infer_from[infer_from.columns[:-1]]  # drop rating

reg = LinearRegression().fit(infer_from.drop('pH', axis=1).values, infer_from['pH'].values)
predicted_ph = reg.predict(no_ph_rows[no_ph_rows.columns[:-1]].drop('pH', axis=1).values)

reg_df = o_df.copy()
reg_df['pH'].loc[destroy_idx] = predicted_ph
experiments['linear_reg_ph_impute'] = reg_df

# fill empties with kmeans means of cluster of missing ones
print('Filling empties with K-Means')
kmeans = KMeans(random_state=42).fit(infer_from.drop('pH', axis=1))

labels = kmeans.predict(infer_from.drop('pH', axis=1))

infer_from['label'] = labels
map = {k: v for k, v in infer_from.groupby('label')['pH'].mean().items()}

pred_labels = kmeans.predict(no_ph_rows[no_ph_rows.columns[:-1]].drop('pH', axis=1))

predictor = lambda l: map[l]
vfunc = np.vectorize(predictor)
k_ph = vfunc(pred_labels)

cluster_df = o_df.copy()
cluster_df['pH'].loc[destroy_idx] = k_ph
experiments['kmean_ph_impute'] = cluster_df

print('Running Experiments')
for i, (k,v) in enumerate(experiments.items()):
    print(f'\nExperiment - {k}')
    testset = X_test_no_ph if i == 1 else X_test

    precision, recall, f1 = run_experiment(v, best_params, testset, y_test)
    print(f'{k} -> Precision: {precision} - Recall: {recall} - F1: {f1}')
