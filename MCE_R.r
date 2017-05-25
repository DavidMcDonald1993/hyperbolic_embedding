
library(igraph)
library(proxy)

mce <- function(x, n, centring){
	
# This code is performs just MCE and ncMCE (both SVD-based)

	#Given a distance or correlation matrix x, it performs Minimum Curvilinear 
	#Embedding (MCE) or non-centred MCE (ncMCE) (coded 24-March-2013 by 
	#Gregorio Alanis-Lobato and checked by Carlo Vittorio Cannistraci)

	#INPUT
	#   x => Distance (example: Euclidean) or distance-adjusted correlation matrix (example: x = 1 - Pearson_correlation)
	#   n => Dimension into which the data is to be embedded
	#   centring => 'yes' if x should be centred or 'no' if not
	#OUTPUT
	#   s => Coordinates of the samples being embedded into the reduced n-dimesional space

	#Make sure the required library 'igraph' is installed and load it
	if(require("igraph")){
		print("igraph has been loaded...");
	} else{
		print("Trying to install igraph...");
		install.packages("igraph");
		if(require("igraph")){
			print("igraph has been installed and loaded...");
		} else{
			stop("Could not install igraph");
		}
	}

	#Make sure the matrix is symmetric
	x <- pmax(x, t(x));

	#Create a graph object out of the adjacency matrix x
	g <- graph.adjacency(x, mode = "undirected", weighted = TRUE);

	#MC-kernel computation
	mst <- minimum.spanning.tree(g);
	kernel <- shortest.paths(mst);

	#Kernel centring
	if(centring == "yes"){
		N <- nrow(kernel);
		J <- diag(N) - (1/N)*matrix(1, N, N); #Form the centring matrix J
		kernel <- (-0.5)*(J %*% kernel^2 %*% J);
	}

	#SVD-based Embedding
	res <- svd(kernel);
	L <- diag(res$d);
	V <- res$v;

	sqrtL <- sqrt(L[1:n, 1:n]);
	V <- V[, 1:n];

	s <- t(sqrtL %*% t(V));

	return(s);
}

# dd <- read.table("Uetz_screen.txt")
# G <- graph.data.frame(dd, directed=FALSE)
G <- graph("Zachary")
# G <- read_graph("embedded_network_69.gml", format="gml")
G_decomposed <- decompose(G)
G <- G_decomposed[[which.max(sapply(G_decomposed, function(i) length(V(i))))]]
G <- simplify(G)

?cluster_edge_betweenness

plot(cluster_fast_greedy(G, ), G)

A <- as.matrix(as_adj(G))

A <- as.matrix(dist(A, method = "cosine"))

d <- 50
X <- mce(x = A, n = d+1, centring = "no")
X <- X[,1:d+1]

D <- as.matrix(dist(X))

H <- graph_from_adjacency_matrix(D, mode="undirected", weighted = T)

H

mst <- minimum.spanning.tree(H)
SP <- distances(mst)
colnames(SP) <- V(G)
rownames(SP) <- V(G)

SP

c <- hclust(as.dist(SP), method="ward.D")
plot(c)

library(NMI)

true_labels <- V(G)$firstlevelcommunity
assignments <- cutree(c, k=9)
NMI(data.frame(node = V(G)$id, label = true_labels), data.frame(node = V(G)$id, assignment = assignments))

max(true_labels)

library(vegan)

iso <- isomap(as.dist(distances(G)), ndim=2, k=5)

plot(iso$points)

x <- iso$points[,1]
y <- iso$points[,2]

theta <- atan2(y, x)
theta[theta < 0] <- theta[theta < 0] + 2*pi

theta <- sort(theta)

theta_norm <- sapply(1:length(theta), function(i) i * 2 * pi / length(theta))
names(theta_norm) <- names(theta)

theta_norm

r <- 10
x_p <- r * cos(theta_norm)
y_p <- r * sin(theta_norm)

plot(x_p, y_p, asp = 1)


