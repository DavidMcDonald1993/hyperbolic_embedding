
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.constraints import unit_norm


# In[2]:

G = nx.read_gml("galFiltered.gml").to_undirected()


# In[3]:

genes_in_network = nx.get_node_attributes(G, "label").values()


# In[4]:

expression_data = pd.read_csv("galExpData.csv")


# In[5]:

expression_data


# In[6]:

genes_in_expression_data = expression_data.loc[:,["GENE"]].as_matrix().flatten()


# In[7]:

genes_in_network = [gene for gene in genes_in_network if gene in genes_in_expression_data]


# In[8]:

expression_data_matrix = expression_data.loc[:,["gal1RGsig", "gal4RGsig", "gal80Rsig"]].as_matrix()


# In[9]:

significance_similarity = cosine_similarity(expression_data_matrix) - np.identity(expression_data_matrix.shape[0])


# In[10]:

significance_similarity = pd.DataFrame(significance_similarity, 
                                       columns=genes_in_expression_data, index=genes_in_expression_data)


# In[11]:

significance_similarity = significance_similarity.loc[genes_in_network, genes_in_network].as_matrix()


# In[12]:

significance_similarity /= significance_similarity.sum(axis=0)


# In[13]:

# subnetwork that is labelled
nodes_of_interest = [k for k, v in nx.get_node_attributes(G, "label").items() if v in genes_in_network]
G = G.subgraph(nodes_of_interest)


# In[14]:

A = nx.adjacency_matrix(G)
W = A.dot(np.diag(1./np.array(nx.degree(G).values())))


# In[15]:

W = pd.DataFrame(W, index=genes_in_network, columns=genes_in_network)

W = W.loc[genes_in_network, genes_in_network].as_matrix()


# In[17]:

# transition matrix
T = 0.5 * np.identity(len(genes_in_network)) + 0.4 * W + 0.1 * significance_similarity


# In[18]:

def matrix_multiply(M, n):
    if n == 0:
        return np.identity(M.shape[0])
    return M.dot(matrix_multiply(M, n - 1))


# In[19]:

target_distributions = matrix_multiply(T, 3)


# In[20]:

target_distributions = target_distributions.transpose()


# In[21]:

N, _ = target_distributions.shape


# In[22]:

inputs = np.identity(N)


# In[23]:

x = Input(batch_shape=(50, N))
y = Dense(256, activation="tanh")(x)
y = Dropout(0.5)(y)
y = Dense(128, activation="tanh")(y)
y = Dropout(0.5)(y)
y = Dense(64, activation="tanh")(y)
y = Dropout(0.5)(y)
embedding = Dense(2, activation="linear")(y)
observation_probabilities = Dense(N, activation="softmax")(embedding)

model = Model(x, observation_probabilities)


# In[24]:

model.compile(loss="kld", optimizer="adam")


# In[25]:

model.fit(inputs, target_distributions, batch_size=50, epochs=50000, verbose=False)


# In[40]:

w = model.layers[-1].get_weights()
weights = w[0]
biases = w[1]


# In[26]:

embedder = Model(x, embedding)


# In[27]:

embeddings = embedder.predict(inputs)


# In[43]:

embeddings = embeddings + weights.transpose()


# In[35]:

s = [10*np.exp(p ** 2 / (2 * 0.5 ** 2)) for p in expression_data_matrix[:,0] ]


# In[36]:

s


# In[44]:

plt.figure(figsize=(15, 15))

# for label, i, j in zip(G.nodes(), embeddings[:, 0], embeddings[:, 1]):
#     plt.annotate(
#         label,
#         xy=(i, j), xytext=(-20, 20),
#         textcoords='offset points', ha='right', va='bottom',
#         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#         arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

nodes = np.array(G.nodes())
for n1, n2 in G.edges():
    i, = np.where(nodes == n1)
    j, = np.where(nodes == n2)
    plt.plot(embeddings[(i, j), 0], embeddings[(i, j), 1], c="k", 
#              linewidth = 3 * np.exp(- np.linalg.norm(S_encoded[i] - S_encoded[j]) ** 2 / (2 * 0.5 ** 2) ))
             linewidth = 0.3)


plt.scatter(embeddings[:,0], embeddings[:,1], s = s)


# In[45]:

prediction = model.predict(inputs)


# In[46]:

prediction[0,]


# In[47]:

target_distributions[0,]


# In[ ]:



