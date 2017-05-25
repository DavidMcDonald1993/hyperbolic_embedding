
library(igraph)
library(vegan)

# g <- graph("zachary")

# edges <- read.table("cites.csv")
# g <- graph.data.frame(edges, directed = F)

g <- read_graph("embedded_polbooks.gml", "gml")

g_decomposed <- decompose(g)
g <- g_decomposed[[which.max(sapply(g_decomposed, function(i) length(V(i))))]]
g <- simplify(g)

nodes <- 1:length(V(g))
assignments <- V(g)$value

nodes

assignments

partition_density <- function(community) {
    
    # number of nodes
    n <- length(V(community))
    
    # number of edges
    m <- length(E(community))
    
    return (n * ((m - (n - 1)) / ((n - 2) * (n - 1))) )
}

score_partition <- function(g, communities) {
 
    # number of nodes in network
    N <- length(V(g))

    # number of communities
    k <- length(communities)

    # weighted patition score for entire network
    return(2 / (sqrt(k) * N) * sum(sapply(communities, partition_density)))

}

# local density

# structural similairty funciton
structureSimilarity <- function(g, nv, nw) {
    return(length(intersect(nv, nw)) / 
          sqrt(length(nv) * length(nw)))
}

# pre compute neighbourhoods
neighbourhoods <- sapply(V(g), function(v) {
    neighborhood(g, v, order=1)
})


# compute similiarity of all nodes 
structureSimilarities <- sapply(neighbourhoods, function(nv) {
    sapply(neighbourhoods, function(nw) {
        structureSimilarity(g, nv, nw)
    })
})


# distance = 1 / similarity
distances <- sapply(1:nrow(structureSimilarities), function(i){
    sapply(1:ncol(structureSimilarities), function(j) {
        if (i == j){
            return (0)
        } else if (structureSimilarities[i,j] == 0){
            return (.Machine$double.xmax)
        } else {
            return (1 / structureSimilarities[i,j])
        }
    })
})

# isomap embedding 
iso <- isomap(dist = as.dist(distances), ndim = 2, k = 5)
X <- iso$points

# distance between all points
d <- as.matrix(dist(X))

#distance cut off value
dc <- 3

# density of node
rhos <- apply(d, 1, function(row) {
    sum(row < dc)
})
names(rhos) <- nodes

nodes

# compute deltas
deltas <- sapply(1:length(rhos), function(i) {
    
    #row in distance matrix of interest
    d2Node <- d[names(rhos)[i],]
    
    if (rhos[i] == max(rhos)) {
        # assign point with moximum density the greatest distance
        return(max(d2Node))
    } else {
        #assign minimum distance of node with greater rho
        return(min(d2Node[rhos > rhos[i]]))
    }
    
}) 
names(deltas) <- nodes

# gammas
gammas <- rhos * deltas
names(deltas) <- nodes

plot(rhos, deltas)
text(rhos, deltas, labels = nodes, pos=4) 

plot(g)

dim(X)

X[27,]

matrix(c(X[27,], X[9,], X[30,]), 3, 2, byrow = T)

library(cluster)
kme <- kmeans(X, centers = matrix(c(X[9,], X[27,], X[30,]), ncol=2, byrow = T))
clusplot(X, kme$cluster, color=TRUE, shade=TRUE,
   labels=2, lines=0, main="Cluster Plot")

label_assignment <- function() {
    
}
