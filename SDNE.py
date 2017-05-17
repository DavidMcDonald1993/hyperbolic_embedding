
# coding: utf-8

# In[ ]:

import numpy as np
import networkx as nx

from gem.embedding.sdne import SDNE
from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.utils import graph_util

# Instatiate the embedding method with hyperparameters
sdne = SDNE(2, 0.1, 0.1, 0.01, 0.01, 2, np.array([10]), 0.9, 1000, 0.01, 100)

# Load graph
# graph = graph_util.loadGraph('gem/data/karate.edgelist')
graph = nx.karate_club_graph()

# Learn embedding - accepts a networkx graph or file with edge list
Y, t = sdne.learn_embedding(graph, edge_f=None, is_weighted=True, no_python=True)

# Evaluate on graph reconstruction
MAP, prec_curv = sdne.evaluateStaticGraphReconstruction(graph, sdne, Y, None)


# In[ ]:



