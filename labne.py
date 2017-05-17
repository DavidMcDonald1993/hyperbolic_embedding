
# coding: utf-8

# In[29]:

from __future__ import division

import numpy as np
from sklearn.metrics import pairwise_distances
import networkx as nx


# In[11]:

A = np.array([[0, 1, 1, 1, 0], [0, 0, 1, 1, 0], [1, 0, 0, 1, 0]])


# In[12]:

A


# In[26]:

K = A + 5 * (1-pairwise_distances(A, metric="cosine"))


# In[18]:

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)


# In[19]:

A


# In[28]:

A[0,1]


# In[24]:

nx.get_node_attributes(G, "club").values()[:10]


# In[31]:

n = nx.number_of_nodes(G)
e = nx.number_of_edges(G)
degrees = nx.degree(G).values()
B = np.array([[A[i,j] - (degrees[i] * degrees[j]) / (2 * e) for j in range(n)] for i in range(n)])


# In[32]:

B


# In[33]:

H = np.zeros((n, 2))
for row in H:
    if np.random.rand() < 0.5:
        row[0] = 1
    else:
        row[1] = 1


# In[36]:

np.trace(np.dot(np.transpose(H), H))


# In[37]:

Q = np.trace(np.dot(np.dot(np.transpose(H), B), H))


# In[38]:

Q


# In[39]:

assignments = nx.get_node_attributes(G, "club").values()
I = np.zeros((n, 2))
for i in range(n):
    if assignments[i] == "Mr. Hi":
        I[i,0] = 1
    else:
        I[i,1] = 1


# In[41]:

np.trace(np.dot(np.dot(np.transpose(I), B), I))


# In[ ]:



