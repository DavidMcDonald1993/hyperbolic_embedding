
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
from scipy.stats import norm
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gml("galFiltered.gml").to_undirected()
G = max(nx.connected_component_subgraphs(G), key=len)

labels = nx.get_node_attributes(G, "label")

genes_in_network = labels.values()

expression_data = pd.read_csv("galExpData.csv")

genes_in_expression_data = expression_data.loc[:,["GENE"]].as_matrix().flatten()

genes_in_network = [gene for gene in genes_in_network if gene in genes_in_expression_data]

# subnetwork that is labelled
nodes_of_interest = [k for k, v in nx.get_node_attributes(G, "label").items() if v in genes_in_network]
G = G.subgraph(nodes_of_interest)

p_values = expression_data.set_index("GENE").loc[genes_in_network,["gal1RGsig", "gal4RGsig", "gal80Rsig"]].as_matrix()

z_values = norm.ppf(1 - p_values)
z_values[z_values < 0] = 0
z_values = z_values[:, 0, np.newaxis]

A = np.array(nx.adjacency_matrix(G).todense())
Z = z_values.dot(z_values.transpose())

A = Z * A

G = nx.from_numpy_matrix(A)
G = max(nx.connected_component_subgraphs(G), key=len)

nodes = G.nodes()

Z = Z[nodes][:, nodes]

labels = {k: v for k,v in labels.items() if k in G.nodes()}
nx.set_node_attributes(G, "label", labels)

N = nx.number_of_nodes(G)

A = np.array(nx.adjacency_matrix(G).todense())
D = A.sum(axis=0)

W = (np.identity(N) + A.dot(np.diag(1./D))) / 2

def matrix_multiply(M, n):
    if n == 0:
        return np.identity(M.shape[0])
    if n % 2 == 0:
        m = matrix_multiply(M, n/2).dot(matrix_multiply(M, n/2))
    else: m = M.dot(matrix_multiply(M, n-1))
    m[m < 0] = 0
    return m / m.sum(axis=0)

# targets = matrix_multiply(W, n=50)
targets = matrix_multiply(W, n=5).transpose()

D_n = targets.sum(axis=0)

sorted_nodes = D_n.argsort()[::-1]
# beta = 1.0 / (2.5 - 1)
# R = 2 * beta * np.log(range(1, N + 1)) + 2 * (1 - beta) * np.log(N + 1)
# R[sorted_nodes] = R


# In[2]:

Z.shape


# In[3]:

sorted_nodes


# In[4]:

def hyperbolic_distance(x1, x2):

    # compute hyperbolic distance 
    delta = np.pi - np.abs(np.pi - np.abs(x1[1] - x2[1]))
    d = np.cosh(x1[0]) * np.cosh(x2[0]) - np.sinh(x1[0]) * np.sinh(x2[0]) * np.cos(delta)
    d = np.maximum(1.0, d)
    d = np.arccosh(d)

    return d

def compute_probabilities(X):
    
    D = np.exp( - np.array([[hyperbolic_distance(i, j) for j in X] for i in X]))
    
    return D / D.sum(axis=1)[:, None]

def kullback_leibler_divergence(y_true, y_pred):
    
    y_true = np.clip(y_true, 1e-8, 1.0)
    y_pred = np.clip(y_pred, 1e-8, 1.0)
    
    return np.sum(np.sum(y_true * np.log(y_true / y_pred), axis=-1))

def distance_loss_function(individual):
    
    # reshape into [r, theta]
    X = individual.reshape(-1, 2)
    
    # probabilities
    P = compute_probabilities(X)
    
    return kullback_leibler_divergence(targets, P)

def popularity_loss_function(individual):
    
    # reshape into [r, theta]
    X = individual.reshape(-1, 2)
    
    # are nodes sorted by their popularity as in the PS model?
    individual_sort = X[:,0].argsort()
    
    # number of nodes that appear in a different order
    return np.array([D_n[i] < D_n[j] for i, j in zip(individual_sort, individual_sort[1:])]).sum()

def enrichment_distance_loss_function(individual):
    
    # reshape into [r, theta]
    X = individual.reshape(-1, 2)
    
    # pairwise  hyperbolic distances multiplied by expression
    return (np.array([[hyperbolic_distance(i, j) for j in X] for i in X]) * Z).sum()

def evaluate(individual):
    # convert to numpy array for evaluation
    individual = np.array(individual)
    return distance_loss_function(individual), popularity_loss_function(individual), enrichment_distance_loss_function(individual)


# In[5]:

#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import json

# import numpy as np

# from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools


# In[6]:

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

BOUND_LOW, BOUND_UP = 0.0, 2 * np.pi

NDIM = N * 2

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)


# In[ ]:

def main(seed=None):
    random.seed(seed)
    
    NGEN = 500
    MU = 100
    CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [1000.0, 1000.0, 1000.0]))

    return pop, logbook
        
if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    
    pop, stats = main()
    # pop.sort(key=lambda x: x.fitness.values)
    
    # print(stats)
    # print("Convergence: ", convergence(pop, optimal_front))
    # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))
    
    # import matplotlib.pyplot as plt
    # import numpy
    
    # front = numpy.array([ind.fitness.values for ind in pop])
    # optimal_front = numpy.array(optimal_front)
    # plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    # plt.scatter(front[:,0], front[:,1], c="b")
    # plt.axis("tight")
    # plt.show()


# In[ ]:

pop.sort(key=lambda x: x.fitness.values)


# In[ ]:

best_individual = np.array(pop[0]).reshape(-1, 2)


# In[ ]:

R = best_individual[:, 0]
theta = best_individual[:, 1]


# In[ ]:

X = np.column_stack([R * np.cos(theta), R * np.sin(theta)])


# In[ ]:

c = ["r" if z else "b" for z in z_values > np.percentile(z_values, 50).flatten()]


# In[ ]:

plt.figure(figsize=(15, 15))

# for label, i, j in zip(genes_in_network, X[:10, 0], X[:10, 1]):
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
    plt.plot(X[(i, j), 0], X[(i, j), 1], c="k", 
             linewidth = 0.3)

plt.scatter(X[:,0], X[:,1], c = c, s = 100)


# In[ ]:



