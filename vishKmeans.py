import pandas as pd

dataset = pd.read_csv('Example.csv')

dataset.head(2)

import matplotlib.pyplot as plt

sat = dataset['Satisfaction']

loy = dataset['Loyalty']

plt.scatter(sat, loy)
plt.xlabel('sat')
plt.ylabel('loy')

# loy = loy * 1000

#dataset['Loyalty'] = loy

dataset


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

data_scaled = sc.fit_transform(dataset)

data_scaled

from sklearn.cluster import KMeans

model = KMeans(n_clusters=4)

model.fit(data_scaled)

pred  = model.fit_predict(data_scaled)

pred

dataset_scaled = pd.DataFrame(data_scaled, columns=['Sat', 'Loy'])

# dataset_scaled

dataset_scaled['cluster name'] = pred

# dataset_scaled

dataset['cluster name'] = pred

plt.scatter(dataset_scaled['Loy'], dataset_scaled['Sat'], c=dataset_scaled['cluster name'])