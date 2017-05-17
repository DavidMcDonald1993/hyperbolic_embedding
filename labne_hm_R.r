
library("devtools")

library("NetHypGeom")

net <- ps_model(N = 500, avg.k = 10, gma = 2.0, Temp = 0.01)

plot_degree_distr(net$network)

plot_hyperbolic_net(network = net$network, nodes = net$polar, node.colour = net$polar$theta)

conn <- get_conn_probs(net = net$network, polar = net$polar, bins = 25)
plot(conn$dist, conn$prob, pch = 16, xlab = "Hyperbolic distance", ylab = "Connection probability")

fit_power_law(degree(net$network))$alpha

# Form vectors of random non-redundant source-target pairs
N <- vcount(net$network)
st <- 1000

# We subtract 1, because the formulae to go from linear upper 
# diagonal indexing to (i,j) are zero-based
k <- sample(N*(N-1)/2, st) - 1
sources <- (N - 2 - floor(sqrt(-8*k + 4*N*(N-1)-7)/2.0 - 0.5))
targets <- (k + sources + 1 - N*(N-1)/2 + (N-sources)*((N-sources)-1)/2)

# Back to 1-based indexing
sources <- sources + 1
targets <- targets + 1

# Analyse the network's navigability
hop.stretch <- greedy_route_packets(net$network, net$polar, sources, targets)

# Compute the fraction of succesfully delivered packets
sum(hop.stretch > 0)/st

# To embed the network using HyperMap, we set LaBNE+HM's window to 2*pi
hm <- labne_hm(net = net$network, Temp = 0.15, k.speedup = 10, w = 2*pi)

# To embed with LaBNE+HM, we reduce HyperMap's search space from 2*pi 
# to a small window of 15 degrees around LaBNE's angles
lh <- labne_hm(net = net$network, Temp = 0.15, k.speedup = 10, w = pi/12)

# Comparison between real and HyperMap-inferred angles and real and LaBNE+HM-inferred angles
plot(net$polar$theta, hm$polar$theta, pch = 16, 
     xlab = "Real angles", ylab = "Inferred angles", main = "HyperMap")
plot(net$polar$theta, lh$polar$theta, pch = 16, 
     xlab = "Real angles", ylab = "Inferred angles", main = "LaBNE+HM")

lh$polar$theta

plot(net$polar$theta, sapply(1:500, function(i) 2 * pi * i / 500), pch = 16, 
     xlab = "Real angles", ylab = "Inferred angles", main = "LaBNE+HM")

plot_hyperbolic_net(network = lh$network, nodes = lh$polar, node.colour = lh$polar$theta)

zach <- graph("Zachary") # the Zachary karate club

plot(zach, vertex.size=10)

lh_zachary <- labne_hm(net = zach, gma = 2.3, Temp = 0.15, k.speedup = 10, w = pi/12)

plot_hyperbolic_net(network = lh_zachary$network, nodes = lh_zachary$polar, node.colour = lh_zachary$polar$theta)

lh_zachary

polar <- lh_zachary$polar

distances <- sapply(1:nrow(polar), function(i) {
    sapply(1:nrow(polar), function(j) {
        hyperbolic_dist(polar[i,], polar[j,])
    })
})

clusters <- hclust(as.dist(distances))
plot(clusters, xlab = "Nodes")

cutree(clusters, k = c(2))

kmea <- kmeans(as.dist(distances), 2)

kmea$cluster

actual_assignments <- data.frame(1:34, c(1,1,1,1,1,1,1,1,1,2,1,1,1,1,2,2,1,1,2,1,2,1,2,2,2,2,2,2,2,2,2,2,2,2))

pred_assignments <- data.frame(1:34, kmea$cluster)

library(NMI)

NMI(actual_assignments, pred_assignments)

shortest_paths <- distances(zach)

mst <- minimum.spanning.tree(zach)

mst_shortest_paths <- distances(zach)

shortest_paths

all(mst_shortest_paths == shortest_paths)

dd <- read.table("reactome_edgelist.txt")
gg <- graph.data.frame(dd, directed=FALSE)

gg_mst <- minimum.spanning.tree(gg)
dist <- distances(gg)
gg_mst_dist <- distances(gg_mst)

all(dist == gg_mst_dist)


