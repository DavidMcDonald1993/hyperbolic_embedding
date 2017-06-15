
# coding: utf-8

# In[5]:

G = nx.DiGraph()
G.add_edge(1, 2)


# In[6]:

np.array(nx.adj_matrix(G).todense())


# In[2]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from sys import stdout


# In[269]:

biogrid_edgelist = np.genfromtxt("BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.4.149.tab2.txt", skip_header=True,
                                usecols=(1,2), dtype=np.int)


# In[273]:

biogrid_edgelist


# In[276]:

np.savetxt(X=biogrid_edgelist, fname="biogrid_edgelist.txt",fmt="%i")


# In[264]:

biogrid_edgelist.shape


# In[2]:

# G = nx.karate_club_graph()
# G = nx.read_gml("embedded_football.gml")
# G = nx.read_gml("dolphins_labelled.gml")
# G = nx.read_gml("embedded_yeast_union.gml")
G = nx.read_edgelist("biogrid_edgelist.txt")
G = max(nx.connected_component_subgraphs(G), key=len)


# In[3]:

N = nx.number_of_nodes(G)


# In[4]:

A = np.array(nx.adjacency_matrix(G).todense())
D = np.diag(A.sum(axis=1))


# In[5]:

# P = np.ones(N) / N
P = np.append(np.ones(1), np.zeros(N - 1))


# In[47]:

loop = 0.0

W = A.dot(np.diag(1./D.diagonal()))
W = loop * np.identity(N) + (1 - loop) * W


# In[48]:

def expansion(M):
    return np.linalg.matrix_power(M, 2)


# In[49]:

def inflation(M, r=2):
    M = M ** r
    return M / M.sum(axis=0)


# In[51]:

M = W

for i in range(5):
    M = expansion(M)
    M = inflation(M, r=2)
    
    stdout.write("\rCompleted iteration {}".format(i))
    stdout.flush()


# In[46]:

M


# In[42]:

assignments = M.argmax(axis=0)
assignments


# In[43]:

len(np.unique(assignments))


# In[32]:

num_assigned = {i: len(np.where(assignments==i)[0]) for i in np.unique(assignments)}


# In[33]:

num_assigned


# In[34]:

nx.draw(G.subgraph([G.nodes()[n] for n in np.where(assignments == assignments[485])[0]]))


# In[249]:

len(np.unique(assignments))


# In[234]:

from sklearn.metrics import normalized_mutual_info_score


# In[235]:

normalized_mutual_info_score(assignments, nx.get_node_attributes(G, "group").values())


# In[119]:

plt.imshow(M)


# In[95]:

dist = np.linalg.matrix_power(W, 3).dot(P)


# In[103]:

[G.node[i]["label"] for i in np.where(dist>0.01)[0]]


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



