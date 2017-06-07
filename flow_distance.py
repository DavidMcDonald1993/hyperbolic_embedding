
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# In[65]:

# G = nx.karate_club_graph()
# G = nx.read_gml("embedded_football.gml")
G = nx.read_gml("embedded_yeast_union.gml")
# assignments = np.array(nx.get_node_attributes(G, "value").values())


# In[66]:

N = nx.number_of_nodes(G)


# In[67]:

A = np.array(nx.adjacency_matrix(G).todense())
D = np.diag(A.sum(axis=1))


# In[68]:

pi = A.sum(axis=1, dtype=np.float32)/A.sum()


# In[69]:

# P = np.ones(N) / N
P = np.append(np.ones(1), np.zeros(N - 1))


# In[70]:

W = A.dot(np.diag(1./D.diagonal()))
W = 0.5 * np.identity(N) + 0.5 * W


# In[88]:

dist = np.linalg.matrix_power(W, 7).dot(P)


# In[89]:

dist[dist>0].shape


# In[60]:

com_prob = {k: 0 for k in np.unique(assignments)}


# In[61]:

for c, p in zip(assignments, dist):
    com_prob[c] += p


# In[62]:

com_prob


# In[63]:

S = np.diag(1./np.sqrt(D.diagonal())).dot(A).dot(np.diag(1./np.sqrt(D.diagonal())))


# In[84]:

l, U = np.linalg.eigh(S)


# In[85]:

l = l[::-1]
U = U[:,::-1]


# In[90]:

l


# In[86]:

pi


# In[87]:

U[:,0]


# In[ ]:



