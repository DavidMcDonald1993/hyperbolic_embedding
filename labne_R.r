
library("devtools")

library("NetHypGeom")

net <- ps_model(N = 500, avg.k = 10, gma = 2.3, Temp = 0.15)

plot_degree_distr(net$network)

compare_clustering_to_er(net$network, 1000, "PS network")

plot_hyperbolic_net(network = net$network, nodes = net$polar, node.colour = net$polar$theta)

conn <- get_conn_probs(net = net$network, polar = net$polar, bins = 25)
plot(conn$dist, conn$prob, pch = 16, xlab = "Hyperbolic distance", ylab = "Connection probability")

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
hm <- labne_hm(net = net$network, gma = 2.3, Temp = 0.15, k.speedup = 10, w = 2*pi)

# To embed with LaBNE+HM, we reduce HyperMap's search space from 2*pi 
# to a small window of 15 degrees around LaBNE's angles
lh <- labne_hm(net = net$network, gma = 2.3, Temp = 0.15, k.speedup = 10, w = pi/12)

# Comparison between real and HyperMap-inferred angles and real and LaBNE+HM-inferred angles
plot(net$polar$theta, hm$polar$theta, pch = 16, 
     xlab = "Real angles", ylab = "Inferred angles", main = "HyperMap")
plot(net$polar$theta, lh$polar$theta, pch = 16, 
     xlab = "Real angles", ylab = "Inferred angles", main = "LaBNE+HM")

plot_hyperbolic_net(network = lh$network, nodes = lh$polar, node.colour = lh$polar$theta)

zach <- graph("Zachary") # the Zachary carate club

plot(zach, vertex.size=10)

lh_zachary <- labne_hm(net = zach, gma = 2.3, Temp = 0.15, k.speedup = 10, w = pi/12)

plot_hyperbolic_net(network = lh_zachary$network, nodes = lh_zachary$polar, node.colour = lh_zachary$polar$theta)

lh_zachary

ba <-  sample_pa(n=500, power=2, m=10,  directed=F)
plot(ba, vertex.size=6, vertex.label=NA)

lh_ba <- labne_hm(net = ba, gma = 2.3, Temp = 0.15, k.speedup = 10, w = pi/12)

plot_hyperbolic_net(network = lh_ba$network, nodes = lh_ba$polar, node.colour = lh_ba$polar$theta)

polar <- lh_ba$polar

distances <- sapply(1:nrow(polar), function(i) {
    sapply(1:nrow(polar), function(j) {
        hyperbolic_dist(polar[i,], polar[j,])
    })
})

clusters <- hclust(as.dist(distances))
plot(clusters, xlab = "Nodes")

cutree(clusters, k = c(25, 24, 20, 15))

cutree(clusters, h = c(25, 24, 20, 15))

ceb <- cluster_louvain(zach)

plot(ceb, zach)

distances <- sapply(1:nrow(lh_zachary$polar), function(i) {
    sapply(1:nrow(lh_zachary$polar), function(j) {
        hyperbolic_dist(lh_zachary$polar[i,], lh_zachary$polar[j,])
    })
})

clusters <- hclust(as.dist(distances))
plot(clusters, xlab = "Nodes")

fit <- cmdscale(as.dist(distances), k=2, eig=TRUE)

fit

plot(fit$point[,1], fit$points[,2], col="blue", lty="solid")
text(fit$point[,1], fit$point[,2], labels=1:34, cex= 0.7, pos=2)

cl <- kmeans(x = fit$points, centers = 2)

cl$cluster

plot(fit$points, col=cl$cluster)

library(NMI)

labels <- c(1,1,1,1,1,1,1,1,1,2,1,1,1,1,2,2,1,1,2,1,2,1,2,2,2,2,2,2,2,2,2,2,2,2)
cbind(1:34, labels)

NMI(cbind(1:34,cl$cluster), cbind(1:34, labels))

typeof(labels)

cl$cluster

cutree(tree = clusters, k=2)

# comparing 2 cluster solutions
library(fpc)
cluster.stats(as.dist(distances), cl$cluster, cutree(tree = clusters, k=2)) 

# K-Means Clustering with 5 clusters
kme <- kmeans(fit$points, 5)

# Cluster Plot against 1st 2 principal components

# vary parameters for most readable graph
library(cluster)
clusplot(fit$points, kme$cluster, color=TRUE, shade=TRUE,
   labels=2, lines=0, main="Cluster Plot")

# Centroid Plot against 1st 2 discriminant functions
library(fpc)
plotcluster(fit$points, kme$cluster) 

edges <- read.table("Y2H_union.txt")

g <- graph.data.frame(edges, directed = F)

g_decomposed <- decompose(g)
g <- g_decomposed[[which.max(sapply(g_decomposed, function(i) length(V(i))))]]
# g <- simplify(g)

transitivity(g)

fit_power_law(degree(g))$alpha

plot_degree_distr(g)

reactome_lh <- labne_hm(net = g)

plot_hyperbolic_net(net=reactome_lh$network, 
                    nodes = reactome_lh$polar, node.colour = reactome_lh$polar$theta)

conn <- get_conn_probs(net = reactome_lh$network, polar = reactome_lh$polar, bins = 15)
plot(conn$dist, conn$prob, pch = 16, xlab = "Hyperbolic distance", ylab = "Connection probability")

emb <- cmdscale(as.dist(distances(g)), k=2, eig=T)

emb$eig

plot(emb$points)

library(GOSemSim)
library(topGO)
library(org.Sc.sgd.db)

scGO <- godata(OrgDb = "org.Sc.sgd.db", keytype = "ORF", ont = "BP")

genes <- V(g)$name

geneSim <- mgeneSim(genes = genes, semData = scGO, measure = "Wang", combine = "BMA")

embeddedGenes <- emb$points[rownames(emb$points) %in% rownames(geneSim),1:6]

dim(embeddedGenes)

# mds
mdsDist <- dist(embeddedGenes)
mdsCl <- hclust(mdsDist)
mdsCophDist <- cophenetic(mdsCl)

# correlation between mds distance and cophenetic distance
cor(mdsDist, mdsCophDist)

# GO term similarity
goDist <- as.dist(1 - geneSim)
goCl <- hclust(goDist)
goCophDist <- cophenetic(goCl)

# correlation between GO dissimilarity and cophenetic distance
cor(goDist, goCophDist)

# correlation between GO dissimilarity and mds cophenetic distance
cor(goDist, mdsCophDist)

neighbourhoods <- sapply(genes, function(v) {
    neighborhood(g, v, order=1)
})

structureSimilarity <- function(g, nv, nw) {
    return(length(intersect(nv, nw)) / 
          sqrt(length(nv) * length(nw)))
}

structureSimilarities <- sapply(neighbourhoods, function(nv) {
    sapply(neighbourhoods, function(nw) {
        structureSimilarity(g, nv, nw)
    })
})

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

library(vegan)

iso <- isomap(as.dist(distances), ndim=10, k=5)

rownames(geneSim)

rownames(iso$points) <- genes

rownames(iso$points)

embeddedGenes <- iso$points[rownames(iso$points) %in% rownames(geneSim), 1:2]

dim(embeddedGenes)

# isomap
isoDist <- dist(embeddedGenes)
isoCl <- hclust(isoDist)
isoCophDist <- cophenetic(isoCl)

# correlation between mds distance and cophenetic distance
cor(isoDist, isoCophDist)

# GO term similarity
goDist <- as.dist(1 - geneSim)
goCl <- hclust(goDist)
goCophDist <- cophenetic(goCl)

# correlation between GO dissimilarity and cophenetic distance
cor(goDist, goCophDist)

# correlation between GO dissimilarity and mds cophenetic distance
cor(goDist, isoCophDist)


