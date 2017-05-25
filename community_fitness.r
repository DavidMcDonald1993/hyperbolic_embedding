
library(igraph)
library(proxy)

community_fitness <- function(A_G, A_H, alpha) {
    
    # edges in community
    d_in <- sum(A_H)
    
    # edges out of community
    d_out <- sum(A_G[!rownames(A_G)%in%rownames(A_H), rownames(A_H)])
    
    if (is.na(d_in / (d_in + d_out) ** alpha)){
        print("NAN")
        print(d_in)
        print(d_out)
        print(A_G[!rownames(A_G)%in%rownames(A_H), rownames(A_H)])
        print(A_H)
        stop()
    }
    
    return(d_in / (d_in + d_out) ** alpha)
}

community_neighbours <- function(A_G, A_H) {
        
    # nodes not in community
    other_nodes <- rownames(A_G)[!rownames(A_G)%in%rownames(A_H)]
    
    if (length(other_nodes) == 1) {
        return(other_nodes)
    }
        
    # connections to H
    C <- A_G[other_nodes, rownames(A_H)]

    # filter C for rows with a connection
    if (class(C) == "matrix") {
        return(rownames(C)[apply(C, 1, function(row) any(row > 0))])        
            
    } else return(names(C)[C > 0]) # H only contains one node so C is a vector
}
                 
node_fitness <- function(A_G, A_H, alpha, v) {
    
    # adjacency matrix of H with v
    A_H_add <- A_G[unique(c(rownames(A_H), v)), unique(c(colnames(A_H), v))]
    
    # adjacency matrix of H without v
    A_H_remove <- as.matrix(A_G[rownames(A_H)[rownames(A_H) != v], colnames(A_H)[colnames(A_H) != v]])
    rownames(A_H_remove) <- rownames(A_H)[rownames(A_H) != v]
    colnames(A_H_remove) <- colnames(A_H)[colnames(A_H) != v]
    
    return(community_fitness(A_G, A_H_add, alpha) - 
          community_fitness(A_G, A_H_remove, alpha))
}

greedy_community_search <- function(A_G, alpha=1.0, num_communities=100, overlaps_allowed=TRUE, min_community_size=3) {
    
    # initialise list of communities dicovered
    communities <- list()
    # discard isolated nodes
    connected_nodes <- rownames(A_G)[apply(A_G, 1, function(row) any(row > 0))]
    # vector of whetehr each node has been asigned to at least oine community
    node_assigned <- rep(FALSE, length(connected_nodes))
    names(node_assigned) <- connected_nodes
    # initiliase community counter
    community <- 1
    # proceed until all nodes are assgined or enough communities have been found
    while (!all(node_assigned) && community < num_communities) {
        
        #select random seed node from uncovered nodes
        r <- sample(1:sum(!node_assigned), 1)
        v <- names(node_assigned)[!node_assigned][r]

        # initialise community (as adjacancy matrix A_H) with seed node
        A_H <- as.matrix(A_G[v, v])
        rownames(A_H) <- v
        colnames(A_H) <- v
        
        # neighbours of community
        neighbours <- community_neighbours(A_G, A_H)

        # main loop: continue until no neiughbours can be found
        while(length(neighbours) > 0) {
            
            # calculate fitness of neighbours
            neighbour_fitnesses <- sapply(neighbours, function(neighbour) node_fitness(A_G, A_H, alpha, neighbour))
            names(neighbour_fitnesses) <- neighbours

            # add best neighbour to H
            if (max(neighbour_fitnesses) > 0) {
                # determine best neighbour
                best_neighbour <- names(neighbour_fitnesses)[which.max(neighbour_fitnesses)]
                # create new adjacency matrix for H
                A_H <- A_G[c(rownames(A_H), best_neighbour), c(colnames(A_H), best_neighbour)]
            } else {
                # No neighbour has positive fitness 
                # TERMINATE
                break
            }

            while (TRUE){
                
                # continuously evaluate fitness of all nodes in H and remove nodes with negative fitness
                node_fitnesses <- sapply(rownames(A_H), function(v) node_fitness(A_G, A_H, alpha, v)) 

                if (any(node_fitnesses < 0)){

                    # remove all nodes with negative fitness
                    A_H_new <- as.matrix(A_G[rownames(A_H)[node_fitnesses >= 0], colnames(A_H)[node_fitnesses >= 0]])
                    rownames(A_H_new) <- rownames(A_H)[node_fitnesses >= 0]
                    colnames(A_H_new) <- colnames(A_H)[node_fitnesses >= 0]

                    # update A_H
                    A_H <- A_H_new

                    # continue if only one node
                    if (length(rownames(A_H)) == 1) break

                } else break # all nodes have positive fitness

                
            } 
                    
            # neighbours of community
            neighbours <- community_neighbours(A_G, A_H)

        }
                    
        # check that community is not a duplicate
        if (any(sapply(communities, function(com) all(rownames(A_H)%in%com)))) break
            
        # assign all nodes in A_H to this community
#         communities[rownames(A_H)] <- community
        communities[[community]] <- rownames(A_H)
                    
        # mark each node in this community as assigned
        node_assigned[rownames(A_H)] <- TRUE
                    
        if (!overlaps_allowed) {
            # remove community from network
            nodes_to_keep <- rownames(A_G)[!rownames(A_G)%in%rownames(A_H)]
            A_G <- A_G[nodes_to_keep, nodes_to_keep]
            if (is.null(dim(A_G))){
                A_G <- matrix(A_G)
                rownames(A_G) <- nodes_to_keep
                colnames(A_G) <- nodes_to_keep
            }
        }            
                                
        #increment community counter
        community <- community + 1
    }     
                                         
     
                                         
    #return community list
    return(communities)
}



increment_consensus_matrix <- function(consensus_matrix, community_list) {

    for (community in community_list) {
        
        # nodes assigned to this community
        if(length(community) == 0) next
            
        # iterate over pairs of these nodes and increase weight in consensus matrix
        for (i in 1:length(community)) {
            n_1 <- community[i]
            for (j in i:length(community)) {
                if (i == j) next
                n_2 <- community[j] 
                    
                # increment consensus matrix
                consensus_matrix[n_1, n_2] <- consensus_matrix[n_1, n_2] + 1
                consensus_matrix[n_2, n_1] <- consensus_matrix[n_2, n_1] + 1                
            }
        }
    } 
        
    return(consensus_matrix)
    
}

adjacency_matrix_to_consensus_matrix <- function(A_G, alpha=1.0, 
                                                 num_repeats=100, num_communities=100, overlaps_allowed=TRUE) {
    
    # run algorithm num_repeats times
    community_assignments <- sapply(1:num_repeats, function(i) 
        greedy_community_search(A_G, alpha=alpha, num_communities=num_communities, overlaps_allowed=overlaps_allowed))
    
    # convert to consensus matrix
    # initialise consesnsus matrix
    consensus_matrix <- matrix(0, nrow=nrow(A_G), ncol=ncol(A_G))
    rownames(consensus_matrix) <- rownames(A_G)
    colnames(consensus_matrix) <- colnames(A_G)
    
    # update consensus matrix
    for (community_assignment in community_assignments) {
        consensus_matrix <- increment_consensus_matrix(consensus_matrix, community_assignment)
    }
        
    # remove any nodes not assigned to community
#     assigned_nodes <- rownames(consensus_matrix)[apply(consensus_matrix, 1, function(row) any(row > 0))]
#     consensus_matrix <- consensus_matrix[assigned_nodes, assigned_nodes]
        
    # normalise
#     consensus_matrix <- consensus_matrix / max(consensus_matrix)
        
    return(consensus_matrix)
}

# read in graph
G <- read.graph("embedded_yeast_uetz.gml", "gml")

# weight by similarity
S1 <- as.matrix(as_adj(G))
S2 <- 1 - as.matrix(dist(S1, method = "cosine"))

# weighting of similarity
w1 <- 5

# create weighted adjacancy matrix
A_G <- S1 + w1 * S2
rownames(A_G) <- V(G)
colnames(A_G) <- V(G)

heatmap(A_G)

# resolution parameter
alpha <- 1.0

# filtering
tau <- 0.0

# convert to consensus matrix
consensus_matrix <- A_G 

# iterate until consensus matrix is all 1s and 0s
# iter <- 0
# while (iter < 2 && !all(consensus_matrix %in% c(0,1))) {
#     # obtain consensus matrix
#     consensus_matrix <- adjacency_matrix_to_consensus_matrix(consensus_matrix, alpha=alpha, num_repeats = 10)
#     # apply filtering
#     consensus_matrix[consensus_matrix < tau] <- 0
#     #increement iteration
#     iter <- iter + 1
# }

# TODO: multiple scales simultaneously
for (iter in 1:3) {
    multi_scale_consensus_matrices <- lapply(c(0.8, 1.0, 1.2, 1.4), function(alpha)
        adjacency_matrix_to_consensus_matrix(consensus_matrix, alpha=alpha, num_repeats = 10))   
        
    consensus_matrix <- matrix(0, nrow=nrow(A_G), ncol=ncol(A_G))
    for (mat in multi_scale_consensus_matrices) {
        consensus_matrix <- consensus_matrix + mat
    }
}


consensus_matrix

heatmap(consensus_matrix)

# cluster based on consensus matrix
assignments <- greedy_community_search(consensus_matrix, alpha=alpha, overlaps_allowed = T)

# convert back to ORF
orfCommunities <- sapply(assignments, function(com) V(induced.subgraph(G, com))$label)
orfCommunities

# plot G coloured by assignment 
plot.igraph(G, vertex.color=assignments)

library(NMI)

true_df <- data.frame(nodes=V(G)$id, true=V(G)$value)
pred_df <- data.frame(nodes=V(G)$id, pred=assignments)
NMI(true_df, pred_df)$value

library(topGO)
library(GOSemSim)
library(GOSim)
library(org.Sc.sgd.db)

scGO <- godata(OrgDb = "org.Sc.sgd.db", keytype = "ORF", ont = "BP")

overlaps <- sapply(orfCommunities, function(i) sapply(orfCommunities, function(j) length(intersect(i, j))))

overlaps

clusterSim <- mclusterSim(orfCommunities, semData = scGO)

clusterSim

setEvidenceLevel(organism = org.Sc.sgdORGANISM, evidences = "all", gomap = org.Sc.sgdGO)
setOntology("BP", loadIC = F,)

# filter out genes not in database
orfCommunities <- sapply(orfCommunities, function(com) com[com%in%keys(org.Sc.sgd.db)])

goEnrichmentResults <- sapply(orfCommunities, function(com) GOenrichment(com, allgenes = keys(org.Sc.sgd.db)))

interClusterSim <- sapply(orfCommunities, function(com){
    sims <- mgeneSim(com, semData = scGO, verbose = F)
    return(mean(sims[upper.tri(sims)]))
})

interClusterSim

GO_A <- mgeneSim(genes = V(G)$label, semData = scGO)

head(GO_A)

filter <- 0.45
GO_F <- GO_A
GO_F[GO_F < filter] <- 0

GO_G <- graph_from_adjacency_matrix(GO_F, weighted = T)

is_connected(GO_G)

edge_density(GO_G)

GO_F

edge_density(G)

GOCommunities <- greedy_community_search(GO_F, alpha = 1.0)

interClusterSimGO <- sapply(GOCommunities, function(com){
    sims <- mgeneSim(com, semData = scGO, verbose = F)
    return(mean(sims[upper.tri(sims)]))
})

GOCommunities[[3]]

interClusterSimGO


