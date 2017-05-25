
# coding: utf-8

# In[1]:

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from gem.utils import graph_util

# Load graph
# graph = graph_util.loadGraph('gem/data/karate.edgelist')
graph = nx.karate_club_graph().to_directed()

# second level similarity
S1 = np.array(nx.adj_matrix(graph).todense())
S2 = cosine_similarity(S1)

S = S1 + 0 * S2

nodes = np.array(graph.nodes())
weights = {(u, v): S[np.where(nodes==u)[0], np.where(nodes==v)[0]][0] for u, v in graph.edges()}

nx.set_edge_attributes(graph, "weight", weights)


# In[23]:

from gem.embedding.gf import GraphFactorization as gf

# Instatiate the embedding method with hyperparameters
em = gf(d=2, max_iter=10000, eta=1e-4, regu=1.0)

# Learn embedding - accepts a networkx graph or file with edge list
Y, t = em.learn_embedding(graph, edge_f=None, is_weighted=True, no_python=True)


# In[17]:

graph.edges(data=True)


# In[18]:

get_ipython().magic(u'pinfo SDNE')


# In[ ]:

from gem.embedding.sdne import SDNE

sdne = SDNE(d=2, beta=1, alpha=1e-2, nu1=1e-3, nu2=1e-3, K=2, n_units=[15,],
            rho=0.3, n_iter=10000, xeta=0.01, n_batch=500)

Y, t = sdne.learn_embedding(graph, edge_f=None, is_weighted=True, no_python=True)


# In[ ]:

colours = np.array(["r" if club ==  "Mr. Hi" else "b" for club in nx.get_node_attributes(graph, "club").values()])


# In[ ]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
plt.scatter(Y[:,0], Y[:,1], c =colours)


# In[ ]:



