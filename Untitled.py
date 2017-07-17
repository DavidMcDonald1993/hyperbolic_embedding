
# coding: utf-8

# In[10]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# In[7]:

X = np.genfromtxt("karate.emb", skip_header=True)


# In[9]:

G = nx.karate_club_graph()


# In[21]:

c = ["r" if v == "Mr. Hi" else "b" for v in nx.get_node_attributes(G, "club").values()]


# In[22]:

plt.scatter(X[:,1], X[:,2], c = [c[x-1] for x in X[:,0].astype(np.int)])


# In[17]:

X[:,0].astype(np.int)


# In[ ]:



