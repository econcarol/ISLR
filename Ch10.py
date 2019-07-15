# ISLR Ch 10 by Carol Cui

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# Q7
USArrests  = pd.read_csv('C:\\Users\\Carol\\Desktop\\USArrests.csv')
state_name = USArrests.iloc[:,0] # abstract a list of state names
# rename index of each row to its corresponding state name
USArrests  = USArrests.rename(index = lambda x: state_name[x]) 
# drop the column of state name from the original data frame
USArrests.drop(USArrests.columns[[0]], axis=1, inplace=True)

scaled_x  = StandardScaler().fit_transform(USArrests) 
scaled_tx = scaled_x.transpose()

Edis  = distance.pdist(scaled_tx, metric='euclidean')
Edis  = distance.squareform(Edis)
Edis2 = Edis**2
corr  = np.corrcoef(scaled_x, rowvar=False)
prop  = (1-corr)/Edis2
print(prop) # 0.01

# ----------------------------------------------------------------------------
# Q8
# (a)
pca       = PCA()
reduced_x = pca.fit_transform(scaled_x)
pve1      = pca.explained_variance_ratio_
print(pve1)

# (b)
pve2 = np.zeros(4)
for i in range(0,4):
    pve2[i] = np.sum(np.matmul(pca.components_[i,:],scaled_tx)**2)/np.sum(scaled_tx**2)
print(pve2)   

# ----------------------------------------------------------------------------
# Q9
# (a)
hc_complete1 = linkage(USArrests, 'complete')

# (b)
res1            = pd.DataFrame(cut_tree(hc_complete1, n_clusters = 3), index=state_name)
res1.index.name = 'states'
res1.rename(columns={0: 'ID'}, inplace=True)

# (c)
hc_complete2    = linkage(scaled_x, 'complete')
res2            = pd.DataFrame(cut_tree(hc_complete2, n_clusters = 3), index=state_name)
res2.index.name = 'states'
res2.rename(columns={0: 'ID'}, inplace=True)

# (d)
res = res1.join(res2, lsuffix='_or', rsuffix='_sd')

# plots
fig1, (ax1, ax2) = plt.subplots(2,1)
ax1.scatter(range(0,len(state_name)), res1)
ax1.set_title('original data')
ax2.scatter(range(0,len(state_name)), res2, color='red')
ax2.set_title('scaled data')

# dendrogram
plt.figure(2)
plt.title('Hierarchical Clustering Dendrogram original data')
plt.xlabel('states')
plt.ylabel('Ecludian distance')
dendrogram(hc_complete1, labels=res1.index, leaf_rotation=90, leaf_font_size=8)
plt.axhline(y=150, c='k', ls='dashed')
plt.show()

plt.figure(3)
plt.title('Hierarchical Clustering Dendrogram scaled data')
plt.xlabel('states')
plt.ylabel('Ecludian distance')
dendrogram(hc_complete2, labels=res2.index, leaf_rotation=90, leaf_font_size=8)
plt.axhline(y=4.45, c='k', ls='dashed')
plt.show()

# Before scaling, cluster sizes are more equal. 
# Scaling puts more states in one particular cluster.  
# Scaling is better cuz crime frequencies and variable units are different. 

# ----------------------------------------------------------------------------
# Q10
# (a)
n = 60
p = 50

np.random.seed(1)
y = np.concatenate((np.repeat(0,20), np.repeat(1,20), np.repeat(2,20)))
x = np.random.rand(n,p)
x[1:20, 1:20]   = x[1:20, 1:20] + 3
x[21:40, 21:40] = x[21:40, 21:40] - 4

# (b)
pca       = PCA()
reduced_x = pca.fit_transform(x)
plt.scatter(reduced_x[:,0], reduced_x[:,1], c=y, cmap=mpl.cm.Paired)
plt.xlabel('Z1')
plt.ylabel('Z2')

# (c)
kmeans3 = KMeans(n_clusters=3, n_init=20, random_state=2).fit(x)
pd.crosstab(index=kmeans3.labels_, columns=y, rownames=['K-Means 3'], colnames=['y'])
# Kmeans correctly identifies 3 clusters.

# (d)
kmeans2 = KMeans(n_clusters=2, n_init=20, random_state=3).fit(x)
pd.crosstab(index=kmeans2.labels_, columns=y, rownames=['K-Means 2'], colnames=['y'])
# Kmeans assign y0 and y2 to the same cluster.

# (e)
kmeans4 = KMeans(n_clusters=4, n_init=20, random_state=4).fit(x)
pd.crosstab(index=kmeans4.labels_, columns=y, rownames=['K-Means 4'], colnames=['y'])
# Kmeans splits y2 into 2 different clusters.

# (f)
kmeans3pca = KMeans(n_clusters=3, n_init=20, random_state=5).fit(reduced_x[:,0:2])
pd.crosstab(index=kmeans3pca.labels_, columns=y, rownames=['K-Means PCA 3'], colnames=['y'])
# Kmeans correctly identifies 3 clusters.

# (g)
scaled_x  = StandardScaler().fit_transform(x) 
kmeans3sd = KMeans(n_clusters=3, n_init=20, random_state=6).fit(scaled_x)
pd.crosstab(index=kmeans3sd.labels_, columns=y, rownames=['K-Means scaled 3'], colnames=['y'])
# Kmeans correctly identifies 3 clusters.

# ----------------------------------------------------------------------------
# Q11
# (a)
df = pd.read_csv('C:\\Users\\Carol\\Desktop\\Ch10Ex11.csv', header=None)
df = df.T

# (b)
# dendrograms
all_hc   = []
linkages = ['complete', 'average', 'single']
for i, link in zip(range(1,4), linkages):
    hc = linkage(y=df, method=link, metric='correlation')
    all_hc.append(hc)
    plt.figure(i)
    plt.title('linkage: %s' %link)
    plt.xlabel('genes')
    plt.ylabel('correlation-based distance')
    dendrogram(hc, labels=df.index, leaf_rotation=90, leaf_font_size=10)

# cluster results
# complete: identify the decreased group correctly
res1 = pd.DataFrame(cut_tree(all_hc[0], n_clusters = 2))
res1.index.name = 'patients'
res1.rename(columns={0: 'ID_complete'}, inplace=True)
res1

# average: identify the decreased group correctly
res2 = pd.DataFrame(cut_tree(all_hc[1], n_clusters = 2))
res2.index.name = 'patients'
res2.rename(columns={0: 'ID_average'}, inplace=True)
res2

# single: fail to identify the decreased group 
res3 = pd.DataFrame(cut_tree(all_hc[2], n_clusters = 2))
res3.index.name = 'patients'
res3.rename(columns={0: 'ID_single'}, inplace=True)
res3

# Yes, my results depend on the type of linkage used. 

# (c)
pca       = PCA()
reduced_x = pca.fit_transform(df)
pve       = pca.explained_variance_ratio_

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(range(0, len(pve)), pve)
ax1.set_xlabel('principal component')
ax1.set_ylabel('PVE')
ax2.scatter(range(0, len(pve)), np.cumsum(pve), color='brown')
ax2.set_xlabel('principal component')
ax2.set_ylabel('cumulative PVE')
# There is an elbow after 1st principal component. 

# conclusion
PC1                    = pd.DataFrame({'gene': range(1, len(df.T)+1), 'loading_vec': pca.components_[0,:]})
PC1['abs_loading_vec'] = abs(PC1['loading_vec'])
PC1_top5               = PC1.sort_values(by=['abs_loading_vec'], ascending=False)['gene'][:5]
print('Top 5 genes that differ the most across 2 groups: %s' %np.array(PC1_top5))
