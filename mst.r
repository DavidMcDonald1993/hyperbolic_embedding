
library(igraph)
library(proxy)

G <- read_graph("embedded_polbooks.gml", "gml")

G

true_communities <- sapply(V(G)$value, function(label){
    if (label == "c") return("red")
    else if (label == "l") return("green")
    else return("blue")    
})

# first and second order similarities
S1 <- 1 - as.matrix(as_adj(G))
S2 <- as.matrix(dist(S1, method = "cosine"))
S3 <- as.matrix(dist(S2, method = "cosine"))

S <- 1 * S1 + 1 * S2 + 1 * S3

H <- graph_from_adjacency_matrix(S, weighted = T, mode="undirected")

mst <- minimum.spanning.tree(H)

mst <- delete_edges(mst, E(mst)[which.max(E(mst)$weight)])

plot.igraph(mst, 
            layout=layout.fruchterman.reingold, vertex.color=components(mst)$membership,
           vertex.label.color=true_communities, vertex.color="white")

data.frame(nodes=1:34, club=V(G)$club, com
           =true_communities)


