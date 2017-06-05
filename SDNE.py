
# coding: utf-8

# In[106]:

import numpy as np
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity

from gem.embedding.sdne import SDNE
from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.utils import graph_util

# Instatiate the embedding method with hyperparameters
sdne = SDNE(d=2, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3, n_units=[50, 15,],
            rho=0.3, n_iter=5, xeta=0.01, n_batch=500 )

# Load graph
# G = graph_util.loadGraph('gem/data/karate.edgelist')
G = nx.karate_club_graph()
# G = nx.read_gml("embedded_football.gml")
# G = nx.read_edgelist("Uetz_screen.txt")
G = max(nx.connected_component_subgraphs(G), key=len).to_directed()
G = nx.convert_node_labels_to_integers(G)

# second level similarity
S1 = np.array(nx.adj_matrix(G).todense())
S2 = cosine_similarity(S1)

# similarity matrix (for weights)
S = S1 + 5 * S2
S /= np.max(S)

nodes = np.array(G.nodes())
weights = {(u, v): S[np.where(nodes==u)[0], np.where(nodes==v)[0]][0] for u, v in G.edges()}

# set weights 
nx.set_edge_attributes(G, "weight", weights)


# In[107]:

# Learn embedding - accepts a networkx graph or file with edge list
Y, t = sdne.learn_embedding(G, edge_f=None, is_weighted=True, no_python=True)


# In[41]:

get_ipython().magic(u'pinfo SDNE')


# In[108]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[109]:

plt.figure(figsize=(15, 15))
plt.scatter(Y[:, 0], Y[:, 1], c=["r" if v=="Mr. Hi" else "b" for v in nx.get_node_attributes(G,  "club").values()], s=100)


# In[105]:

Y


# In[84]:

from sklearn.manifold import MDS


# In[85]:

mds = MDS(n_components=2, dissimilarity="precomputed", metric=True)


# In[88]:

St = mds.fit_transform(1 - S)


# In[92]:

plt.figure(figsize=(15, 15))
plt.scatter(St[:, 0], St[:, 1], 
            c=["r" if v=="Mr. Hi" else "b" for v in nx.get_node_attributes(G,  "club").values()], s=100)


# In[ ]:



