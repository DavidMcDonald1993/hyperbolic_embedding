
library(igraph)
library(proxy)
library(NMI)

library(topGO)
library(GOSemSim)
library(GOSim)
library(org.Sc.sgd.db)

module_fitness <- function(community, sample=F) {
    k <- length(community)
    Z_A <- sum(z_score[community]) / sqrt(k)
    if (sample) return (Z_A)
    return((Z_A - means[k]) / standard_deviatations[k])
}

community_fitness <- function(A_G, community, alpha) {
    
#     # edges in community
#     c <- A_G[community, community]
#     d_in <- sum(c[upper.tri(c, diag = T)])
    
#     # edges out of community
#     d_out <- sum(A_G[!rownames(A_G)%in%community, community])
    
#     if (d_in + d_out == 0){
#         print (community)
#         stop()
#     }
    
#     return(d_in / (d_in + d_out) ** alpha)
    
    return(module_fitness(community))
}

community_neighbours <- function(A_G, community) {
        
    # nodes not in community
    other_nodes <- rownames(A_G)[!rownames(A_G) %in% community]
        
    # connections to H
    C <- A_G[other_nodes, community]

    # filter C for rows with a connection
    if (class(C) == "matrix") {
        return(rownames(C)[apply(C, 1, function(row) any(row != 0))])             
    } else return(names(C)[C != 0]) # H only contains one node so C is a vector
}
                 
node_fitness <- function(A_G, community, alpha, v) {
    
    # add v to A_H
    community_add <- unique(c(community, v))
    
    # fitness of community with v
    fitness_add <- community_fitness(A_G, community_add, alpha)
    
    # remove v from A_H
    community_remove <- community[community != v]
    
    if (length(community_remove) == 0) 
        fitness_remove <- 0 
    else {
        # fitness of community without v
        fitness_remove <- community_fitness(A_G, community_remove, alpha)
    }
    return(fitness_add - fitness_remove)
}
    
probabilistic_neighbour_selection <- function(neighbour_fitnesses, random=F) {
    
    # return best neighbour
    if (!random) return(names(neighbour_fitnesses)[which.max(neighbour_fitnesses)])
        
    # compute probability of selection
    probs <- neighbour_fitnesses
    probs[probs < 0] <- 0
    probs <- probs / sum(probs)
        
    return(sample(names(probs), 1, prob = probs))
    
}

greedy_community_search <- function(A_G, alpha=1.0, max_iter=100, overlaps_allowed=TRUE, min_community_size=3) {
    
    # initialise list of communities dicovered
    communities <- list()
    # discard isolated nodes
#     connected_nodes <- rownames(A_G)[apply(A_G, 1, function(row) any(row > 0))]
    connected_nodes <- rownames(A_G)
#     print(sprintf("connectd nodes: %s", connected_nodes))
    # vector of whether each node has been asigned to at least one community
    node_assigned <- rep(FALSE, length(connected_nodes))
    names(node_assigned) <- connected_nodes
    # proceed until all nodes are assgined or enough communities have been found
    iter <- 1    
    while (!all(node_assigned) && iter < max_iter) {
        
        #select random seed node from uncovered nodes
        r <- sample(1:sum(!node_assigned), 1)
        v <- names(node_assigned)[!node_assigned][r]
        
        # mark as assigned
        node_assigned[v] <- TRUE

        # initialise community with seed node
        community <- v
        
        # neighbours of community
        neighbours <- community_neighbours(A_G, community)

        # continue until no neiughbours can be found
        while(length(neighbours) > 0) {
            
            # calculate fitness of neighbours
            neighbour_fitnesses <- sapply(neighbours, 
                                          function(neighbour) node_fitness(A_G, community, alpha, neighbour))
            names(neighbour_fitnesses) <- neighbours
                                              
              tryCatch({
                  if (max(neighbour_fitnesses) > 0) {
                      
                  }
              }, error = function(e){
                  print(neighbour_fitnesses)
                  print(neighbours)
                  print(community)
                  stop()
              })

            # add best neighbour to H
            if (max(neighbour_fitnesses) > 0) {
                
                # determine best neighbour
#                 best_neighbour <- names(neighbour_fitnesses)[which.max(neighbour_fitnesses)]
                chosen_neighbour <- probabilistic_neighbour_selection(neighbour_fitnesses, random = T)
                # add ot community
                community <- c(community, best_neighbour)
                
            } else break # no neighbour has positive fitness 
                
            # continuously evaluate fitness of all nodes in H and remove nodes with negative fitness
            node_fitnesses <- sapply(community, function(v) node_fitness(A_G, community, alpha, v)) 

            while (any(node_fitnesses < 0)){
                
                # remove all nodes with negative fitness
                community <- community[node_fitnesses >= 0]

                # break if only one node
#                 if (length(rownames(A_H)) == 1) break
                    
                # recalulate node fitnesses    
                node_fitnesses <- sapply(community, function(v) node_fitness(A_G, community, alpha, v)) 
                
            } 
                    
            # neighbours of community
            neighbours <- community_neighbours(A_G, community)

        }
                    
        # check that community is not a subset of another discovered community
        subset <- any(sapply(communities, function(com) 
            length(intersect(com, community)) == length(community)))
            
        # ensure community is large enough
        if (!subset && length(community) > min_community_size){
            
            # check that community is not super set of another discovered community (keep largest community)
            if (length(communities) > 0) {
                
                superset <- which(sapply(communities, function(com)
                length(intersect(com, community)) == length(com)))
                
                # remove all communities that are subsets of this community
                if (length(superset > 0)) 
                    communities <- communities[-superset]
                
            }
            

            # assign all nodes in A_H to this community
            communities <- c(communities, list(community))

            # mark each node in this community as assigned
            node_assigned[community] <- TRUE

            if (!overlaps_allowed) {
                # remove community from network
                nodes_to_keep <- rownames(A_G)[!rownames(A_G)%in%community]
                A_G <- A_G[nodes_to_keep, nodes_to_keep]
                if (is.null(dim(A_G)))
                    A_G <- matrix(A_G)
                rownames(A_G) <- nodes_to_keep
                colnames(A_G) <- nodes_to_keep
            }           
        }
        
        # increment iteration
        iter  <- iter + 1
    }     
                                         
    #return community list
    return(communities)
}



increment_consensus_matrix <- function(consensus_matrix, community_list) {

    # iterate over list of communities
    for (community in community_list) {
        
        if(length(community) == 0) next
            
        # iterate over pairs of nodes and increase the pairwise weight in consensus matrix
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
                                                 num_repeats=100, max_iter=100, 
                                                 overlaps_allowed=TRUE, normalise=FALSE, filter="50%") {
    
    # run algorithm num_repeats times
    community_assignments <- lapply(1:num_repeats, function(i) 
        greedy_community_search(A_G, alpha=alpha, max_iter=max_iter, overlaps_allowed=overlaps_allowed))
    
    # convert to consensus matrix
    # initialise consensus matrix
    consensus_matrix <- matrix(0, nrow=nrow(A_G), ncol=ncol(A_G))
    rownames(consensus_matrix) <- rownames(A_G)
    colnames(consensus_matrix) <- colnames(A_G)
    
    # update consensus matrix
    for (community_assignment in community_assignments) {
        consensus_matrix <- increment_consensus_matrix(consensus_matrix, community_assignment)
    }
        
    # normalisation 
    if (normalise) {
        consensus_matrix <- consensus_matrix / num_repeats
    }    
    
    # filter consensus matrix
    consensus_matrix[consensus_matrix < quantile(consensus_matrix)[filter]] <- 0
        
    return(consensus_matrix)
}

matrix_multiply <- function(M, n) {
    if (n==0) return(diag(nrow=nrow(M)))
    return(M%*%matrix_multiply(M, n-1))
}

random_module_sample <- function(A_G, k) {
    
    module <- sample(rownames(A_G), 1)
    
    while (length(module) < k) {
        
        neighbours <- community_neighbours(A_G, module)
        
        while (length(neighbours) == 0) {
            module <- sample(rownames(A_G), 1)
            neighbours <- community_neighbours(A_G, module)
        }
        
        module <- c(module, sample(neighbours, 1))
        
    }
    
    return(module_fitness(module, sample=T))
    
}

precompute_random_sample_fitnesses <- function(A_G, num_repeats=10, k=20) {
    return(sapply(1:num_repeats, function(r)
        sapply(1:k, function(ki) random_module_sample(A_G, ki))))
}

# read in graph
# df <- read.table("biogrid_edgelist.txt")
# G <- graph.data.frame(df, directed = F)
G <- read.graph("galFiltered.gml", "gml")
# G <- read.graph("reactome_edgelist.txt", format = "ncol", directed = F)
# G <- simplify(G)

expressions <- read.table("galExpData.csv", sep = ",", header = T)

p_values <- expressions[,"gal1RGsig"]
names(p_values) <- expressions[,"GENE"]

z_score <- qnorm(1 - p_values)

genes <- V(G)$label[V(G)$label%in%names(z_score)]

z_score <- z_score[genes]

A <- as.matrix(as_adj(G))
rownames(A) <- V(G)$label
colnames(A) <- V(G)$label

A <- A[genes, genes]

D <- diag(1/igraph::degree(G, mode="all"))
rownames(D) <- V(G)$label
colnames(D) <- V(G)$label

D <- D[genes, genes]

L <- as.matrix(laplacian_matrix(G))
rownames(L) <- V(G)$label
colnames(L) <- V(G)$label

L <- L[genes, genes]

# weight by similarity
S1 <- A
# S2 <- 1 - as.matrix(dist(S1, method = "cosine"))
S2 <- 0

# weighting of second order similarity
w1 <- 0

# create weighted adjacancy matrix
A_G <- S1 + w1 * S2
# rownames(A_G) <- V(G)$label
# colnames(A_G) <- V(G)$label

# normalise similarity matrix
# A_G <- A_G / max(A_G)

# # filter out nodes with no annotations
# nodes_to_keep <- rownames(A_G)[rownames(A_G) %in% names(z_score)]

# # multiply by expression values (hadamand product)
# A_G <- A_G[nodes_to_keep, nodes_to_keep] * Z[nodes_to_keep, nodes_to_keep]

# # normalise A_G
# A_G <- A_G / max(A_G)

k <- 50

sample_fitnesses <- precompute_random_sample_fitnesses(A_G, num_repeats = 100, k = k)
means <- apply(sample_fitnesses, 1, mean)
names(means) <- 1:k
standard_devations <- apply(sample_fitnesses, 1, sd)
names(standard_devations) <- 1:k

mean_file <- "means.rda"
sd_file <- "sd.rda"

if (!file.exists(mean_file)) {
    
    save(means, file=mean_file)
    save(standard_deviations, file=sd_file)
    print("saved files")
    
} else {
    load(mean_file)
    load(sd_file)
    print("loaded files")
}

# resolution parameter(s)
alphas <- c(1.0)

# number of repeats
num_repeats <- 10

# number of iterations
num_iter <- 1

# filtering
tau <- 0.1

# weight decay
lambda <- 0

# convert to consensus matrix
consensus_matrix <- A_G 

for (iter in 1:num_iter) {
    
    # obtain consensus matrix for each resolution
    multi_scale_consensus_matrices <- lapply(alphas, function(alpha)
        adjacency_matrix_to_consensus_matrix(consensus_matrix, alpha=alpha, 
                                             num_repeats = num_repeats, overlaps_allowed = T, 
                                             normalise = F, filter="0%"))   
        
    # initialise consensus matrix
    c_m <- matrix(0, nrow=nrow(A_G), ncol=ncol(A_G))
    rownames(c_m) <- rownames(A_G)
    colnames(c_m) <- colnames(A_G)
        
    # sum consensus matrices for all scales
    for (mat in multi_scale_consensus_matrices) {
        c_m <- c_m + mat
    }
        
    # normalise consensus matrix
    c_m  <- c_m / (length(alphas) * num_repeats)

    # update consensus matrix
    consensus_matrix <- lambda * consensus_matrix + (1 - lambda) * c_m
}

heatmap(consensus_matrix)

# cluster based on consensus matrix
communities <- greedy_community_search(consensus_matrix, max_iter = 1000, 
                                       alpha = 1.0, overlaps_allowed = T, min_community_size = 3)

coms <- 1:length(communities)
names(communities) <- coms

# invert (for NMI)
assignments <- numeric(length = length(V(G)))
names(assignments) <- V(G)$label
for (i in 1:length(communities)) {
    for (node in communities[[i]]) {
        assignments[node] <- i
    }
}

# assignments <- assignments[names(assignments) %in% keys(org.Sc.sgd.db)]

communities

lengths(communities)

module_scores <- sapply(communities, function(com) module_fitness(com))
module_scores

plot.igraph(induced.subgraph(G, vids = V(G)[V(G)$label %in% communities[[12]]]),)

# count intercommunity edges
inter_community_edges <- function(A_G, c1, c2) {
    return(sum(A_G[c1, c2]))
}

edges <- sapply(coms, function(i) sapply(coms, function(j) {
        if(i == j) return (0)
        inter_community_edges(A_G, communities[[i]], communities[[j]])
    }))
rownames(edges) <- coms
colnames(edges) <- coms

sum(edges)

# true_df <- data.frame(node_id=V(G)$id, module=V(G)$club)
# pred_df <- data.frame(node_id=V(G)$id, module=assignments)
# NMI(true_df, pred_df)$value

# prepare godata object for cluster functional similarity
scGO <- godata(OrgDb = "org.Sc.sgd.db", keytype = "ORF", ont = "BP")

# filter out genes not in database
orfCommunities <- sapply(communities, function(com) com[com%in%keys(org.Sc.sgd.db)])

# compute functional similarity of communities
clusterSim <- mclusterSim(orfCommunities, semData = scGO)
rownames(clusterSim) <- coms
colnames(clusterSim) <- coms
diag(clusterSim) <- 0

# filter out < 0.5
clusterSim[clusterSim < 0.5] <- 0

# cluster network based on GO similairity
go_communities <- greedy_community_search(clusterSim,
                                          max_iter = 100, alpha=1.0, overlaps_allowed = F, min_community_size=0)

go_communities

# go_assignments <- go_assignments[rev(order(lengths(go_assignments)))]

# reverse
go_assignments <- numeric(length = length(communities))
names(go_assignments) <- coms
for (i in 1:length(go_communities)) {
    for (com in go_communities[[i]]) {
        go_assignments[com] <- i
    }
}
# go_assignments_ <- components(I)$membership

go_assignments

colors <- rainbow(length(go_communities))
names(colors) <- 1:length(go_communities)

colors

I <- graph_from_adjacency_matrix(clusterSim, mode = "undirected", weighted = T)

plot.igraph(I, vertex.color=colors[go_assignments], edge.label=E(I)$weight)

E <- graph_from_adjacency_matrix(edges, mode = "undirected", weighted = T)

plot.igraph(E, vertex.size=lengths(communities), 
            vertex.color=colors[go_assignments], edge.label=E(E)$weight)

# plot some communities
communities_of_interest <- c(which.max(module_scores))
nodes_of_interest <- unlist(communities[communities_of_interest])

communities_of_interest

node_ids <- V(G)[V(G)$label %in% nodes_of_interest]
sub_G <- induced.subgraph(graph = G, vids = node_ids)
plot.igraph(sub_G, 
#             vertex.color=colors[go_assignments[assignments[nodes_of_interest]]])
            vertex.color=assignments[nodes_of_interest], edge.label=E(sub_G)$weight)

# initialise GoSim for most representative terms
setEvidenceLevel(organism = org.Sc.sgdORGANISM, evidences = "all", gomap = org.Sc.sgdGO)
setOntology("BP", loadIC = F,)

# perform GO term enrichment with topGO
goEnrichmentResults <- sapply(orfCommunities, function(com) 
    GOenrichment(genesOfInterest = com, allgenes = keys(org.Sc.sgd.db), cutoff = 0.05))

rownames(goEnrichmentResults) <- c("GO Terms", "p-values", "genes")

go_terms <- goEnrichmentResults["GO Terms",]
pvalues <- goEnrichmentResults["p-values",]
genes <- goEnrichmentResults["genes",]

go_gene_clusters <- lapply(1:max(go_assignments_), function(i) 
    unlist(orfCommunities[names(go_assignments_)[go_assignments_ == i]]))

# go term enrichment for go clusters
go_gene_enrichment_clusters <- sapply(go_gene_clusters, function(clust) 
    GOenrichment(genesOfInterest = clust, allgenes = keys(org.Sc.sgd.db), cutoff = 0.05))

rownames(go_gene_enrichment_clusters) <- c("GO Terms", "p-values", "genes")
gene_go_terms <- go_gene_enrichment_clusters["GO Terms",]
gene_pvalues <- go_gene_enrichment_clusters["p-values",]
gene_genes <- go_gene_enrichment_clusters["genes",]

get_parent_term <- function(terms) {
    parent_term <- getMinimumSubsumer(terms[1], terms[2])
    
    for (term in terms) {
        parent_term <- getMinimumSubsumer(parent_term, term)
    }
    return(parent_term)
}

most_general_terms <- sapply(gene_pvalues, function(terms) get_parent_term(names(terms)))

most_general_terms <- sapply(gene_pvalues, function(i) names(i)[which.max(i)])

most_general_terms

library(ReactomePA)

entrez_communities <- sapply(go_gene_clusters, function(orfs) {
    conversion <- select(org.Sc.sgd.db, keys = orfs, columns="ENTREZID")[,"ENTREZID"]
    return(conversion[!is.na(conversion)])
})

all_genes_entrez <- select(org.Sc.sgd.db, keys=keys(org.Sc.sgd.db), columns="ENTREZID", keytype="ORF")[,"ENTREZID"]
all_genes_entrez <- all_genes_entrez[!is.na(all_genes_entrez)]

pathway_enrichment_results <- sapply(entrez_communities, function(com)
                                     enrichPathway(gene = com, organism = "yeast", universe = all_genes_entrez))

pathway_enrichment_results

go <- select(GO.db, keys = most_general_terms, columns = c("GOID", "TERM", "DEFINITION"))

graphics.off()

# dev.new(width = 20, height = 20)
plot(E, 
     vertex.label=select(GO.db, 
                keys = sapply(pvalues, function(i) names(i)[which.min(i)]), columns = "TERM")[,"TERM"],
            vertex.size=2*lengths(communities), vertex.color=colors[go_assignments_], edge.label=E(E)$weight)
# legend("bottomleft", legend=go[,"TERM"], col='black', pch=21, pt.bg=colors, text.col=1)

# weight communities by internal edges
diag(edges) <- sapply(communities, function(com) sum(A_G[com, com][upper.tri(A_G[com, com], diag = T)]))

# run again at higher level 
high_order_consensus_matrix <- adjacency_matrix_to_consensus_matrix(edges, overlaps_allowed = T, filter = "0%", normalise = F,
                                                                    alpha = 1.0, num_repeats = 10, max_iter = 1000)

# cluster based on consensus matrix
high_order_communities <- greedy_community_search(high_order_consensus_matrix, max_iter = 1000, 
                                       alpha = 1.0, overlaps_allowed = T, min_community_size = 3)

high_order_coms <- 1:length(high_order_communities)
names(high_order_communities) <- high_order_coms

# invert (for NMI)
high_order_assignments <- numeric(length = nrow(high_order_consensus_matrix))
names(assignments) <- 1:nrow(high_order_consensus_matrix)
for (i in 1:length(high_order_communities)) {
    for (node in high_order_communities[[i]]) {
        high_order_assignments[node] <- i
    }
}

high_order_communities

high_order_coms <- 1:length(high_order_communities)
high_order_edges <- sapply(high_order_coms, function(i) sapply(high_order_coms, function(j) {
    if(i == j) return (0)
    inter_community_edges(edges, high_order_communities[[i]], high_order_communities[[j]])
    }))
rownames(high_order_edges) <- high_order_coms
colnames(high_order_edges) <- high_order_coms

high_order_edges

sum(high_order_edges)

high_order_orf_communities <- sapply(high_order_communities, function(com) unlist(orfCommunities[com]))

lengths(high_order_orf_communities)

high_order_orf_communities

# compute functional similarity of communities
high_order_clusterSim <- mclusterSim(high_order_orf_communities, semData = scGO)
rownames(high_order_clusterSim) <- high_order_coms
colnames(high_order_clusterSim) <- high_order_coms
diag(high_order_clusterSim) <- 0

# filter out < 0.5
high_order_clusterSim[high_order_clusterSim < 0.65] <- 0

high_order_clusterSim

high_order_go_assignments <- greedy_community_search(high_order_clusterSim,
    max_iter = 1000, alpha=1.0, overlaps_allowed = T, min_community_size=0)

high_order_go_assignments

# reverse
high_order_go_assignments_ <- numeric(length = length(high_order_communities))
names(high_order_go_assignments_) <- high_order_coms
for (i in 1:length(high_order_go_assignments)) {
    for (node in high_order_go_assignments[[i]]) {
        high_order_go_assignments_[node] <- i
    }
}

high_order_colors <- rainbow(length(high_order_go_assignments))

J <- graph_from_adjacency_matrix(high_order_clusterSim, mode = "undirected", weighted = T)

plot.igraph(J, vertex.color=high_order_colors[high_order_go_assignments_], edge.label=E(J)$weight)

F <- graph_from_adjacency_matrix(high_order_edges, mode = "undirected", weighted = T)

plot.igraph(F, vertex.size=lengths(high_order_orf_communities)/3, 
            vertex.color=high_order_colors[high_order_go_assignments_], edge.label=E(F)$weight)

high_order_orf_communities[[9]]


