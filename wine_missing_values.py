import random
import pandas as pd

from sklearn.cluster import KMeans

df = pd.read_csv('./winequality-red.csv')

experiments = {}

no_ph = df.copy()
no_ph = no_ph.drop(['pH'], axis=1)

experiments['no_ph'] = no_ph

# destroy 33% of pH column
destroy_idx = random.sample(range(0, len(df)), len(df) // 3)
bad_ph = df.index.isin(destroy_idx)
df['pH'].iloc[destroy_idx] = 0
infer_from = df[~bad_ph]
no_ph_rows = df[bad_ph]


# fill empties with mean
mean_ph = infer_from['pH'].mean()
mean_df = df.copy()
mean_df['pH'].iloc[destroy_idx] = mean_ph

experiments['mean_ph'] = mean_df

# fill empties with kmeans means of cluster of missing ones
kmeans = KMeans(n_clusters=2, random_state=0).fit(infer_from.values)
no_ph_rows = no_ph_rows.drop(['pH'], axis=1)
no_ph_rows = no_ph_rows[no_ph_rows.columns[:-1]] # drop rating too (aposteriori)

# infer ph

# fill empties with regression
