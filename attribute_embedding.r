
library(igraph)

G <- as.undirected(read.graph("galFiltered.gml", "gml"))

expressions <- read.table("galExpData.csv", sep = ",", header = T)

p_values <- expressions[,"gal1RGsig"]
names(p_values) <- expressions[,"GENE"]

z_score <- qnorm(1 - p_values)

genes <- V(G)$label[V(G)$label%in%names(z_score)]

z_score <- z_score[genes]

z_m <- z_score %*% t(z_score)
rownames(z_m) <- colnames(z_m)

exp_z <- exp(z_m)

exp_z <- exp_z / rowSums(exp_z)

A <- as_adjacency_matrix(G)

W <- 0.4 * diag(nrow(A)) + 0.5 * A %*% diag(1/degree(G)) + 0.1 * t(exp_z)

matrix_multiply <- function(M, n) {
    if (n==0) return(diag(nrow=nrow(M)))
    return(M%*%matrix_multiply(M, n-1))
}

T <- matrix_multiply(W, 5)

rownames(T) <- V(G)$label
colnames(T) <- V(G)$label
T <- T[rownames(exp_z), colnames(exp_z)]

lambda <- 0.5

TZ <- (1 - lambda) * T + lambda *  t(exp_z)

TZ <- TZ / colSums(as.matrix(TZ))

TZ <- t(as.matrix(TZ))

write.csv(TZ, file="lambda=05.csv")


