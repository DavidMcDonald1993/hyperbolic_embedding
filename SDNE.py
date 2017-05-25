
# coding: utf-8

# In[1]:

import numpy as np
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity

from gem.embedding.sdne import SDNE
from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.utils import graph_util

# Instatiate the embedding method with hyperparameters
sdne = SDNE(d=2, beta=1, alpha=1e-2, nu1=1e-3, nu2=1e-3, K=2, n_units=[256,],
            rho=0.3, n_iter=500, xeta=0.01, n_batch=500)

# Load graph
# G = graph_util.loadGraph('gem/data/karate.edgelist')
# G = nx.karate_club_graph().to_directed()
# G = nx.read_gml("embedded_football.gml").to_directed()
G = nx.read_edgelist("Uetz_screen.txt")
G = max(nx.connected_component_subgraphs(G), key=len).to_directed()

# second level similarity
S1 = np.array(nx.adj_matrix(G).todense())
S2 = cosine_similarity(S1)

# similarity matrix (for weights)
S = S1 + 5 * S2

nodes = np.array(G.nodes())
weights = {(u, v): S[np.where(nodes==u)[0], np.where(nodes==v)[0]][0] for u, v in G.edges()}

# set weights 
nx.set_edge_attributes(G, "weight", weights)

# Learn embedding - accepts a networkx graph or file with edge list
Y, t = sdne.learn_embedding(G, edge_f=None, is_weighted=True, no_python=True)


# In[3]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[7]:

nx.get_node_attributes(G, "value").values()


# In[6]:

plt.figure(figsize=(15, 15))
plt.scatter(Y[:, 0], Y[:, 1], c=nx.get_node_attributes(G, "value").values(), s=100)


# In[5]:

Y


# In[ ]:



