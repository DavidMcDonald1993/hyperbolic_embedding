import argparse

import numpy as np
import networkx as nx
import pandas as pd

from sys import stdout

from functools import partial
# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool

from sklearn.metrics import normalized_mutual_info_score as NMI

import powerlaw

# import matplotlib.pyplot as plt
# clip value 
clip_value = 1e-8

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def compute_L_G(A, P):
	
	# clip to avoid error
	A_clip = np.clip(A, a_min=clip_value, a_max=1-clip_value)
	
	# return (A_clip * np.log(P) + (1 - A_clip) * np.log(1 - P)).mean()
	return (A_clip * np.log(P) + (1 - A_clip) * np.log(1 - P)).sum()

def compute_L_X(X, Q, attribute_type):

	if attribute_type != "binary":
		# return (0.5 * (X - Q) ** 2).mean()
		return (0.5 * (X - Q) ** 2).sum()
	
	# clip to avoid error
	X_clip = np.clip(X, a_min=clip_value, a_max=1-clip_value)
	
	# return (X_clip * np.log(Q) + (1 - X_clip) * np.log(1 - Q)).mean()
	return (X_clip * np.log(Q) + (1 - X_clip) * np.log(1 - Q)).sum()

def compute_likelihood(A, X, N, K, R, thetas, M, W, lamb_F, lamb_W, alpha, attribute_type):
	
	_, _, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)

	# likelihood of G
	L_G = compute_L_G(A, P)

	# l1 penalty term
	l1_F = lamb_F * np.linalg.norm(F, axis=0, ord=1).sum()


	F = np.column_stack([F, np.ones(N)])

	Q = compute_Q(F, W, attribute_type)
	
	# likelihood of X
	L_X = compute_L_X(X, Q, attribute_type)
	
	# l1
	l1_W = lamb_W * np.linalg.norm(W, axis=0, ord=1).sum()
	
	# overall likelihood
	likelihood = (1 - alpha) * L_G + alpha * L_X - l1_F - l1_W
	
	return L_G, L_X, l1_F, l1_W, likelihood

def hyperbolic_distance(R, thetas, M):
	delta_theta = np.pi - abs(np.pi - abs(thetas[:,None] - M[:,1]))
	x = np.cosh(R[:,None]) * np.cosh(M[:,0]) - np.sinh(R[:,None]) * np.sinh(M[:,0]) * np.cos(delta_theta)
	h = np.arccosh(x)
	return delta_theta, x, h

def compute_F(h, M):
	F = np.exp(- h ** 2 / (2 * M[:,2] ** 2))
	return F

def compute_P(F):
	P = 1 - np.exp(-F.dot(F.T))
	return np.clip(P, a_min=clip_value, a_max=1-clip_value)

def compute_Q(F, W, attribute_type):
	Q = F.dot(W.T)
	if attribute_type != "binary":
		return Q
	Q = sigmoid(Q)
	return np.clip(Q, a_min=clip_value, a_max=1-clip_value)

def compute_Q_u(F_u, W, attribute_type):
	Q_u = F_u.dot(W.T)
	if attribute_type != "binary":
		return Q_u
	Q_u = sigmoid(Q_u)
	return np.clip(Q_u, a_min=clip_value, a_max=1-clip_value)

def compute_Q__k(F, W__k, attribute_type):
	Q__k = F.dot(W__k)
	if attribute_type != "binary":
		return Q__k
	Q__k = sigmoid(Q__k)
	return np.clip(Q__k, a_min=clip_value, a_max=1-clip_value)

def update_theta_u(u, N, K, C, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F
	delta_theta, x, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)
	
	# partial delta theta partial theta
	# partial_delta_theta_u_partial_theta_u = np.array([-np.sign(np.pi - abs(thetas[u] - M[c,1])) *\
	#  -np.sign(thetas[u] - M[c,1]) * 1 for c in range(C)])
	partial_delta_theta_u_partial_theta_u = -np.sign(np.pi - abs(thetas[u] - M[:,1])) *\
	 -np.sign(thetas[u] - M[:,1]) * 1 
	
	# parital h partial theta
	# partial_x_u_partial_delta_theta_u = np.diag([np.sinh(R[u]) * np.sinh(M[c,0]) * np.sin(delta_theta[u,c])
	# 	for c in range(C)])
	partial_x_u_partial_delta_theta_u = np.diag(np.sinh(R[u]) * np.sinh(M[:,0]) * np.sin(delta_theta[u,:]))

	# partial_h_u_partial_x_u = np.diag([1 / np.sqrt(x[u, c] ** 2 - 1) for c in range(C)])
	partial_h_u_partial_x_u = np.diag(1 / np.sqrt(x[u, :] ** 2 - 1))
	 
	# partial F partial theta
	# partial_F_u_partial_h_u = np.diag([-h[u, c] / M[c,2] ** 2 * F[u, c]
	# 	for c in range(C)])
	partial_F_u_partial_h_u = np.diag(-h[u, :] / M[:, 2] ** 2 * F[u, :])

	partial_P_u_partial_F_u = np.array([[F[v, c] * np.exp(-F[u].dot(F[v])) for c in range(C)]
		for v in range(N)])
	
	# partial L_G 
	# partial_L_G_u_partial_P_u = 1.0 / N *\
	partial_L_G_u_partial_P_u =\
	np.array([A[u, v] / P[u, v] - (1 - A[u, v]) / (1 - P[u, v])
		for v in range(N)])

	partial_L_G_u_partial_F_u = partial_L_G_u_partial_P_u\
	.dot(partial_P_u_partial_F_u)

	partial_F_u_partial_theta_u = partial_F_u_partial_h_u\
	.dot(partial_h_u_partial_x_u)\
	.dot(partial_x_u_partial_delta_theta_u)\
	.dot(partial_delta_theta_u_partial_theta_u)

	partial_L_G_u_partial_theta_u = partial_L_G_u_partial_F_u\
	.dot(partial_F_u_partial_theta_u)

	partial_l1_F_u_partial_F_u = np.sign(F[u])

	# partial L_X
	F_u = np.append(F[u], 1)
	Q_u = compute_Q_u(F_u, W, attribute_type)
	# partial_L_X_u_partial_Q_u = 1.0 / K * np.array([X[u,k] / Q_u[k] -\
	# 	(1 - X[u, k]) / (1 - Q_u[k]) for k in range(K)])
	# partial_Q_u_partial_F_u = np.array([[Q_u[k] * (1 - Q_u[k]) * W[k, c] 
	# 	for c in range(C)] for k in range(K)])
	# partial_L_X_u_partial_F_u = partial_L_X_u_partial_Q_u.dot(partial_Q_u_partial_F_u)

	# partial_L_X_u_partial_F_u  = 1.0 / K *\
	partial_L_X_u_partial_F_u  =\
	np.array([(X[u] - Q_u).dot(W[:,c]) for c in range(C)])

	partial_L_X_u_partial_theta_u = partial_L_X_u_partial_F_u.dot(partial_F_u_partial_theta_u)

	# print (1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
	# 	- lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)

	return (1 - alpha) * partial_L_G_u_partial_theta_u + alpha * partial_L_X_u_partial_theta_u\
		- lamb_F * partial_l1_F_u_partial_F_u.dot(partial_F_u_partial_theta_u)

# def compute_delta_theta(N, C, K, A, X, R, thetas, M, W, alpha, lamb_F):
	
# 	# compute F
# 	delta_theta = np.pi - abs(np.pi - abs(thetas[:,None] - M[:,1]))
# 	x = np.cosh(R[:,None]) * np.cosh(M[:,0]) - np.sinh(R[:,None]) * np.sinh(M[:,0]) * np.cos(delta_theta)
# 	h = np.arccosh(x)
# 	F = np.exp(- h ** 2 / (2 * M[:,2] ** 2))
# 	P = 1 - np.exp(-F.dot(F.T))
	
# 	# partial delta theta partial theta
# 	partial_delta_theta_partial_theta = np.array([[[-np.sign(np.pi - abs(thetas[u] - M[c,1])) *\
# 	 -np.sign(thetas[u] - M[c,1]) * 1 if u_prime == u else 0\
# 	 for u_prime in range(N)] for c in range(C)] for u in range(N)])
	
# 	# partial h partial theta
# 	partial_x_partial_delta_theta = np.array([[[[np.sinh(R[u]) * np.sinh(M[c,0]) * np.sin(delta_theta[u,c])\
# 		if u_prime==u and c_prime==c else 0\
# 		for c_prime in range(C)] for u_prime in range(N)] for c in range(C)] for u in range(N)])

# 	partial_h_partial_x = np.array([[[[1 / np.sqrt(x[u, c] ** 2 - 1) if u_prime==u and c_prime==c else 0\
# 		for c_prime in range(C)] for u_prime in range(N)] for c in range(C)] for u in range(N)])
	 
# 	# partial F partial theta
# 	partial_F_partial_h = np.array([[[[-h[u, c] / M[c,2] ** 2 * F[u, c] if u_prime==u and c_prime==c else 0\
# 		for c_prime in range(C)] for u_prime in range(N)] for c in range(C)] for u in range(N)])

# 	partial_P_partial_F = np.array([[[[F[v, c] * np.exp(-F[u].dot(F[v]))\
# 	 if u_prime==u else F[u, c] * np.exp(-F[u].dot(F[v])) if u_prime==v else 0\
# 		for c in range(C)] for u_prime in range(N)] for v in range(N)] for u in range(N)])
	
# 	# partial L_G
# 	partial_L_G_partial_P = 1 / N**2 *np.array([[A[u,v] / P[u,v] - (1 - A[u,v]) / (1 - P[u,v])\
# 		for v in range(N)] for u in range(N)])

# 	partial_L_G_partial_F = np.tensordot(partial_L_G_partial_P,\
# 	partial_P_partial_F)

# 	partial_F_partial_theta = np.tensordot(np.tensordot(np.tensordot(partial_F_partial_h,\
# 	partial_h_partial_x),\
# 	partial_x_partial_delta_theta),\
# 	partial_delta_theta_partial_theta)

# 	partial_L_G_u_partial_theta_u = np.tensordot(partial_L_G_partial_F,\
# 	partial_F_partial_theta)

# 	partial_l1_F_partial_F = np.sign(F)

# 	# partial L_X
# 	F = np.column_stack([F, np.ones(N)])
# 	Q = compute_Q(N, F, W)
# 	partial_L_X_partial_Q = 1 / (N*K) * np.array([[X[u, k] / Q[u, k] - (1 - X[u,k]) / (1 - Q[u, k])
# 		for k in range(K)] for u in range(N)])
# 	partial_Q_partial_F = np.array([[[[Q[u,k] * (1 - Q[u,k]) * W[k, c] if u_prime==u else 0
# 		for c in range(C)] for u_prime in range(N)] for k in range(K)] for u in range(N)])
# 	partial_L_X_partial_F = np.tensordot(partial_L_X_partial_Q, partial_Q_partial_F)

# 	partial_L_X_partial_theta = np.tensordot(partial_L_X_partial_F, partial_F_partial_theta)

# 	return (1 - alpha) * partial_L_G_partial_theta + alpha * partial_L_X_partial_theta\
# 		- lamb_F * np.tensordot(partial_l1_F_partial_F, partial_F_partial_theta)

def update_community_r_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F
	delta_theta, x, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)

	# partial h
	# partial_x_c_partial_r_c = np.array([np.cosh(R[u]) * np.sinh(M[c, 0]) -\
	# 				  np.sinh(R[u]) * np.cosh(M[c, 0]) * np.cos(delta_theta[u, c]) for u in range(N)])
	partial_x_c_partial_r_c = np.cosh(R) * np.sinh(M[c, 0]) -\
					  np.sinh(R) * np.cosh(M[c, 0]) * np.cos(delta_theta[:, c])

	# partial_h_c_partial_x_c = np.diag([1 / np.sqrt(x[u, c] ** 2 - 1) for u in range(N)])
	partial_h_c_partial_x_c = np.diag(1 / np.sqrt(x[:, c] ** 2 - 1))
	
	# partial F
	# partial_F_c_partial_h_c = np.diag([ - h[u, c] / M[c, 2] ** 2 * F[u, c] for u in range(N)]) 
	partial_F_c_partial_h_c = np.diag(-h[:, c] / M[c, 2] ** 2 * F[:, c]) 
	
	# partial L_G TODO
	partial_P_partial_F_c = np.array([[[F[v, c] * np.exp( - F[u].dot(F[v])) if u_prime == u\
	else F[u, c] * np.exp( - F[u].dot(F[v])) if u_prime == v else 0\
	for u_prime in range(N)] for v in range(N)] for u in range(N)])

	# partial_L_G_partial_P = 1.0 / N**2 * \
	partial_L_G_partial_P = \
	np.array([[A[u,v] / P[u,v] - (1 - A[u,v]) / (1 - P[u,v])\
		for v in range(N)] for u in range(N)])

	partial_L_G_c_partial_F_c = np.tensordot(partial_L_G_partial_P, partial_P_partial_F_c)
	
	partial_F_c_partial_r_c = partial_F_c_partial_h_c\
	.dot(partial_h_c_partial_x_c)\
	.dot(partial_x_c_partial_r_c)

	partial_L_G_c_partial_r_c = partial_L_G_c_partial_F_c.dot(partial_F_c_partial_r_c)

	partial_l1_F_c_partial_F_c = np.sign(F[:,c])
	
	# partial L_x 
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)

	# partial_L_X_c_partial_Q = 1.0 / (N*K) * np.array([[X[u, k] / Q[u, k] - (1 - X[u, k]) / (1 - Q[u, k])\
	# 	for k in range(K)] for u in range(N)])
	# partial_Q_partial_F_c = np.array([[[Q[u,k] * (1 - Q[u,k]) * W[k,c] if u_prime == u else 0\
	# 	for u_prime in range(N)] for k in range(K)] for u in range(N)])
	# partial_L_X_c_partial_F_c = np.tensordot(partial_L_X_c_partial_Q, partial_Q_partial_F_c)

	# partial_L_X_c_partial_F_c = 1.0 / (N * K) *\ 
	partial_L_X_c_partial_F_c = \
	np.array([(X[u] - Q[u]).dot(W[:,c]) for u in range(N)])

	partial_L_X_c_partial_r_c = partial_L_X_c_partial_F_c.dot(partial_F_c_partial_r_c)

	# print eta * ((1 - alpha) * partial_L_G_c_partial_r_c)

	return (1 - alpha) * partial_L_G_c_partial_r_c + alpha * partial_L_X_c_partial_r_c\
	 - lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_r_c)

def update_community_theta_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F
	delta_theta, x, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)
	
	# partial delta theta
	# partial_delta_theta_c_partial_theta_c = np.array([-np.sign(np.pi - abs(thetas[u] - M[c,1]))\
	#  * -np.sign(thetas[u] - M[c,1]) * -1 for u in range(N)])
	partial_delta_theta_c_partial_theta_c = -np.sign(np.pi - abs(thetas - M[c,1]))\
	 * -np.sign(thetas - M[c,1]) * -1
	
	# partial h
	# partial_x_c_partial_delta_theta_c = np.diag([np.sinh(R[u]) * np.sinh(M[c,0]) * np.sin(delta_theta[u,c])\
	# 	for u in range(N)])
	partial_x_c_partial_delta_theta_c = np.diag(np.sinh(R) * np.sinh(M[c,0]) * np.sin(delta_theta[:,c]))

	# partial_h_c_partial_x_c = np.diag([1 / np.sqrt(x[u, c] ** 2 - 1) for u in range(N)])
	partial_h_c_partial_x_c = np.diag(1 / np.sqrt(x[:, c] ** 2 - 1))
	
	# partial F
	# partial_F_c_partial_h_c = np.diag([ - h[u, c] / M[c, 2] ** 2 * F[u, c] for u in range(N)]) 
	partial_F_c_partial_h_c = np.diag(-h[:, c] / M[c, 2] ** 2 * F[:, c]) 

	# partial L_G 
	partial_P_partial_F_c = np.array([[[F[v, c] * np.exp( - F[u].dot(F[v])) if u_prime == u\
	else F[u, c] * np.exp( - F[u].dot(F[v])) if u_prime == v else 0\
	for u_prime in range(N)] for v in range(N)] for u in range(N)])

	# partial_L_G_partial_P = 1.0 / N**2 *\
	partial_L_G_partial_P =\
	np.array([[A[u,v] / P[u,v] - (1 - A[u,v]) / (1 - P[u,v])\
		for v in range(N)] for u in range(N)])

	partial_L_G_c_partial_F_c = np.tensordot(partial_L_G_partial_P,partial_P_partial_F_c)

	partial_F_c_partial_theta_c = partial_F_c_partial_h_c\
	.dot(partial_h_c_partial_x_c)\
	.dot(partial_x_c_partial_delta_theta_c)\
	.dot(partial_delta_theta_c_partial_theta_c)

	partial_L_G_c_partial_theta_c = partial_L_G_c_partial_F_c.dot(partial_F_c_partial_theta_c)

	partial_l1_F_c_partial_F_c = np.sign(F[:,c])
	
	# LX
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)

	# partial_L_X_c_partial_Q = 1.0 / (N*K) * np.array([[X[u, k] / Q[u, k] - (1 - X[u, k]) / (1 - Q[u, k])\
	# 	for k in range(K)] for u in range(N)])
	# partial_Q_partial_F_c = np.array([[[Q[u,k] * (1 - Q[u,k]) * W[k,c] if u_prime == u else 0\
	# 	for u_prime in range(N)] for k in range(K)] for u in range(N)])
	# partial_L_X_c_partial_F_c = np.tensordot(partial_L_X_c_partial_Q, partial_Q_partial_F_c)

	# partial_L_X_c_partial_F_c = 1.0 / (N * K) *\
	partial_L_X_c_partial_F_c = \
	np.array([(X[u] - Q[u]).dot(W[:,c]) for u in range(N)])

	partial_L_X_c_partial_theta_c = partial_L_X_c_partial_F_c.dot(partial_F_c_partial_theta_c)

	return (1 - alpha) * partial_L_G_c_partial_theta_c + alpha * partial_L_X_c_partial_theta_c\
		- lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_theta_c)

def update_community_sd_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F, attribute_type):
	
	# compute F

	delta_theta, x, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)
	
	# partial F
	# partial_F_c_partial_sd_c = np.array([h[u, c] ** 2 / M[c, 2] ** 3 * F[u,c] for u in range(N)])
	partial_F_c_partial_sd_c = h[:, c] ** 2 / M[c, 2] ** 3 * F[:,c]

	# partial L_G 
	partial_P_partial_F_c = np.array([[[F[v, c] * np.exp( - F[u].dot(F[v])) if u_prime == u\
	else F[u, c] * np.exp( - F[u].dot(F[v])) if u_prime == v else 0\
	for u_prime in range(N)] for v in range(N)] for u in range(N)])

	# partial_L_G_partial_P = 1.0 / N**2 *\
	partial_L_G_partial_P = \
	np.array([[A[u,v] / P[u,v] - (1 - A[u,v]) / (1 - P[u,v])\
		for v in range(N)] for u in range(N)])

	partial_L_G_c_partial_F_c = np.tensordot(partial_L_G_partial_P,partial_P_partial_F_c)


	partial_L_G_c_partial_sd_c = partial_L_G_c_partial_F_c.dot(partial_F_c_partial_sd_c)

	partial_l1_F_c_partial_F_c = np.sign(F[:,c])
	
	# partial L_x 
	F = np.column_stack([F, np.ones(N)])
	Q = compute_Q(F, W, attribute_type)

	# partial_L_X_c_partial_Q = 1.0 / (N*K) * np.array([[X[u, k] / Q[u, k] - (1 - X[u, k]) / (1 - Q[u, k])\
	# 	for k in range(K)] for u in range(N)])
	# partial_Q_partial_F_c = np.array([[[Q[u,k] * (1 - Q[u,k]) * W[k,c] if u_prime == u else 0\
	# 	for u_prime in range(N)] for k in range(K)] for u in range(N)])
	# partial_L_X_c_partial_F_c = np.tensordot(partial_L_X_c_partial_Q, partial_Q_partial_F_c)

	# partial_L_X_c_partial_F_c = 1.0 / (N * K) *\
	partial_L_X_c_partial_F_c = \
	np.array([(X[u] - Q[u]).dot(W[:,c]) for u in range(N)])

	partial_L_X_c_partial_sd_c = partial_L_X_c_partial_F_c.dot(partial_F_c_partial_sd_c)

	return (1 - alpha) * partial_L_G_c_partial_sd_c + alpha * partial_L_X_c_partial_sd_c\
		- lamb_F * partial_l1_F_c_partial_F_c.dot(partial_F_c_partial_sd_c)


def update_W_k(k, N, K, C, X, W, F, lamb_W, alpha, attribute_type):
	
	# kth row of W
	W_k = W[k]
	
	# kth column of Q
	Q__k = compute_Q__k(F, W_k, attribute_type)
	# Q = compute_Q(F, W, attribute_type)

	# partial_L_X_k_partial_Q_k = 1.0 / N * np.array([X[u, k] / Q[u, k] - (1 - X[u, k]) / (1 - Q[u, k])
	# 	for u in range(N)])
	# partial_Q_k_partial_W_k = np.array([[Q[u, k] * (1 - Q[u, k]) * F[u, c] 
	# 	for c in range(C+1)] for u in range(N)])
	# partial_L_X_k_partial_W_k = partial_L_X_k_partial_Q_k.dot(partial_Q_k_partial_W_k)

	# partial_L_X_k_partial_W_k = 1.0 / N *\
	partial_L_X_k_partial_W_k = \
	np.array([(X[:, k] - Q__k).dot(F[:, c]) for c in range(C + 1)])

	partial_l1_W_k_partial_W_k = np.sign(W_k)
	
	return alpha * partial_L_X_k_partial_W_k - lamb_W * partial_l1_W_k_partial_W_k

def estimate_T():
	'''
	TODO
	'''
	return 0.1

def estimate_gamma(degrees):
	result = powerlaw.Fit(degrees)
	return result.power_law.alpha

def preprocess_G(gml_file, gamma, T):

	G = nx.read_gml(gml_file)
	G = max(nx.connected_component_subgraphs(G), key=len)

	N = nx.number_of_nodes(G)

	degree_dict = nx.degree(G)
	
	# node id
	order_of_appearance = np.array(sorted(degree_dict, key=degree_dict.get, reverse=True))
	nodes = np.array(G.nodes())
	# node index
	order_of_appearance = np.concatenate([np.where(nodes==n)[0] for n in order_of_appearance])

	degrees = np.array(degree_dict.values())

	# PS model parameters -- to estimate in real world network
	m = degrees.mean() / 2
	if T == None:
		stdout.write("T is not given, estimating it")
		T = estimate_T()
	if gamma == None:
		stdout.write("gamma is not given, estimating it")
		gamma = estimate_gamma(degrees)
	beta = 1 / (gamma - 1)

	stdout.write("m={}, T={}, gamma={}, beta={}\n".format(m, T, gamma, beta))

	# determine radial coordinates of nodes
	R = 2 * beta * np.log(range(1, N + 1)) + 2 * (1 - beta) * np.log(N) 
	R = R[order_of_appearance]

	# observed adjacency matrix
	A = np.array(nx.adjacency_matrix(G).todense())

	return N, R, A, order_of_appearance

def preprocess_X(attribute_file):
	attr_df = pd.read_csv(attribute_file, index_col=0,)
	return attr_df.values

def preprocess_true_communities(true_community_file):
	if true_community_file == None:
		return None
	community_df = pd.read_csv(true_community_file, index_col=0,)
	return community_df.values

# def generate_X(N, K, attribute_file, order_of_appearance):

# 	attribute_df = pd.read_csv(attribute_file, sep=" ", index_col=0, dtype=np.float)

# 	attribs = attribute_df.loc[sorted(order_of_appearance),:].iloc[:,0]

# 	# artificially create binary attributes from radial coordinates
# 	X = np.zeros((N, K))

# 	for i, att in enumerate(attribs):
		
# 		id = att * K / attribs.max()
		
# 		p = np.array([np.exp(-(id-x)**2 / (2 * (5.0)**2) ) for x in range(K)])
		
# 		r = np.random.rand(K)
		
# 		X[i][r < p] = 1
		
# 	return X, attribs.values

def initialize_matrices(N, C, K, R):

	sigma = R.max()
	community_radii = R.mean()
	noise = 1e-2
	# community matrix M
	M = np.zeros((C, 3))
	# centre radii
	M[:,0] = np.random.normal(size=(C,), loc=community_radii, scale=noise)
	# center angular coordinate
	M[:,1] = np.random.rand(C) * 2 * np.pi - np.pi
	# M[:,1] = np.random.normal(size=(C,), scale=noise)
	# M[:,1] = np.random.normal(loc = np.arange(C) * 2 * np.pi / C, scale=noise)
	# M[:,1] -= M[:,1].mean()
	# community standard deviations
	M[:,2] = np.random.normal(size=(C,), loc=sigma, scale=noise)

	stdout.write("Initialized M to:\n")
	stdout.write("{}\n".format(M)) 

	# initialise logistic weights
	W = np.random.normal(size=(K, C + 1), scale=noise)

	h = np.sqrt(-2 * sigma ** 2 * np.log( np.sqrt( -np.log(0.5) / C)))

	theta_targets = (np.cosh(R) * np.cosh(community_radii) - np.cosh(h)) / \
		(np.sinh(R) * np.sinh(community_radii))
	theta_targets[theta_targets<-1] = -1
	theta_targets[theta_targets>1] = 1
	# print theta_targets
	theta_targets = np.arccos(theta_targets)

	# print theta_targets
	# theta_targets = 0

	# angular co-ordinates of nodes
	# thetas = np.random.normal(size=(N,), loc=0, scale=noise)
	thetas = np.random.rand(N) * 2 * np.pi - np.pi

	_, _, h = hyperbolic_distance(R, thetas, M)
	F = compute_F(h, M)
	P = compute_P(F)

	return thetas, M, W

def train(A, X, N, K, C, R, thetas, M, W, 
	eta=1e-2, alpha=0.5, lamb_F=1e-2, lamb_W=1e-2, 
	num_processes=5, num_epochs=0, true_communities=None, attribute_type="binary"):

	L_G, L_X, l1_F, l1_W, loss = compute_likelihood(A, X, N, K, R, thetas, M, W, 
		lamb_F=lamb_F, lamb_W=lamb_W, alpha=alpha, attribute_type=attribute_type)
	# alpha = L_X / (L_G + L_X)
	stdout.write("alpha={}, L_G={}, L_X={}, l1_F={}, l1_W={}, total_loss={}\n".format(alpha, L_G, L_X, l1_F, l1_W, loss))

	pool = Pool(num_processes)

	# delta_thetas = np.zeros(thetas.shape)
	# delta_M = np.zeros(M[:,0].shape)
	# delta_W = np.zeros(W.shape)

	for e in range(num_epochs):
		
		# node_order = np.random.permutation(N)
		# node_order = np.arange(N)
		# for u in node_order:
		# 	thetas[u] += eta * update_theta_u(u, N, C, A, X, R, thetas, M, W, alpha, lamb_F)


		delta_thetas = np.array(pool.map(partial(update_theta_u, 
			N=N, K=K, C=C, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
			alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type),
			range(N)))

		# norm = np.linalg.norm(delta_thetas)
		# if norm > 1:
		# 	delta_thetas /= norm
		thetas += eta * delta_thetas

		# print thetas.max()
		# print thetas.min()

		
		# community_order = np.random.permutation(C)
		# for c in community_order:
		# 	M[c, 0] += eta * update_community_r_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F)

		delta_M = np.array(pool.map(partial(update_community_r_c, 
			N=N, K=K, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
			alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type), range(C)))

		# norm = np.linalg.norm(delta_M)
		# if norm > 1:
		# 	delta_M /= norm
		M[:,0] += eta * delta_M

		# community_order = np.random.permutation(C)
		# for c in community_order:
		# 	M[c, 1] += eta * update_community_theta_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F)


		delta_M = np.array(pool.map(partial(update_community_theta_c, 
			N=N, K=K, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
			alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type), range(C)))

		# norm = np.linalg.norm(delta_M)
		# if norm > 1:
		# 	delta_M /= norm
		M[:,1] += eta * delta_M

		# community_order = np.random.permutation(C)
		# for c in community_order:
		# 	M[c, 2] += eta * update_community_sd_c(c, N, K, A, X, R, thetas, M, W, alpha, lamb_F)


		delta_M = np.array(pool.map(partial(update_community_sd_c, 
			N=N, K=K, A=A, X=X, R=R, thetas=thetas, M=M, W=W, 
			alpha=alpha, lamb_F=lamb_F, attribute_type=attribute_type), range(C)))

		# norm = np.linalg.norm(delta_M)
		# if norm > 1:
		# 	delta_M /= norm
		M[:,2] += eta * delta_M

		# community_order = np.random.permutation(C)
		# for c in community_order:
		# 	M[c, 1] = update_community_theta_c(c, N, A, X, R, thetas, M, W, eta, alpha, lamb_F)

		# community_order = np.random.permutation(C)
		# for c in community_order:
		# 	M[c, 2] = update_community_sd_c(c, N, A, X, R, thetas, M, W, eta, alpha, lamb_F)
		
		stdout.write("M=\n")
		stdout.write("{}\n".format(M))
		
		_, _, h = hyperbolic_distance(R, thetas, M)
		F = compute_F(h, M)
		community_predictions = F.argmax(axis=1)
		F = np.column_stack([F, np.ones(N)])

		delta_W = np.array(pool.map(partial(update_W_k, 
			N=N, K=K, C=C, X=X, F=F, W=W, 
			alpha=alpha, lamb_W=lamb_W, attribute_type=attribute_type), range(K)))

		# attribute_order = np.random.permutation(K)
		# for k in attribute_order:
		# 	W[k] += eta * update_W_k(k, N, C, X, W, F, lamb_W, alpha)
		
		W += eta * delta_W

		L_G, L_X, l1_F, l1_W, loss = compute_likelihood(A, X, N, K, R, thetas, M, W, 
			lamb_F=lamb_F, lamb_W=lamb_W, alpha=alpha, attribute_type=attribute_type)
		# alpha = L_X / (L_G + L_X)
		stdout.write("epoch={}, alpha={}, L_G={}, L_X={}, l1_F={}, l1_W={}, total_loss={}\n".format(e, alpha, L_G, L_X, l1_F, l1_W, loss) )

		if true_communities != None:
			# NMI
			stdout.write("NMI: {}\n".format(NMI(true_communities, community_predictions)))

		stdout.flush()

	pool.close()
	pool.join()

	return thetas, M, W

# def draw_network(N, C, R, thetas, M):

# 	_, _, h = hyperbolic_distance(R, thetas, M)
# 	F = compute_F(h, M)
# 	assignments = F.argmax(axis=1)
# 	assignment_strength = np.array([F[i, assignments[i]] for i in range(N)])

# 	node_cartesian = np.column_stack([R * np.cos(thetas), R * np.sin(thetas)])
# 	community_cartesian = np.column_stack([M[:,0] * np.cos(M[:,1]),
# 	M[:,0] * np.sin(M[:,1])])

# 	plt.figure(figsize=(15, 15))
# 	plt.scatter(node_cartesian[:,0], node_cartesian[:,1], c=assignments, s=100*assignment_strength)
# 	plt.scatter(community_cartesian[:,0], community_cartesian[:,1], c=np.arange(C), s=100)
# 	plt.scatter(community_cartesian[:,0], community_cartesian[:,1], c = "k", s=25)
# 	plt.show()


def parse_args():

	parser = argparse.ArgumentParser(description="Embed complex network to hyperbolic space.")
	parser.add_argument("gml_file", metavar="gml_file_path",
	                    help="path of gml file of network")
	parser.add_argument("attribute_file", metavar="attribute_file", 
	                    help="path of attribute file")
	parser.add_argument("num_communities", metavar="C", type=int,
                    help="number of communities")
	parser.add_argument("--T", dest="T", type=np.float,
                    help="network temperature (if this is not given, it is estimated)", default=None)
	parser.add_argument("--gamma", dest="gamma", type=np.float,
                    help="network scaling exponent (if this is not given, it is estimated)",
                    default=None)
	parser.add_argument("--attribute_type", dest="attribute_type", 
                    help="type of attribute (default is binary)", default="binary")
	parser.add_argument("--num_epochs", dest="num_epochs", type=int,
                    help="number of epochs to train for (default is 10000)", default=10000)
	parser.add_argument("--eta", dest="eta", type=np.float,
                    help="learning rate (default is 0.01)", default=1e-2)
	parser.add_argument("--lamb_F", dest="lamb_F", type=np.float,
                    help="l1 penalty on F (default is 0.01) (default is 1e-2)", default=1e-2)
	parser.add_argument("--lamb_W", dest="lamb_W", type=np.float,
                    help="l1 penalty on W (default is 0.01) (default is 1e-2)", default=1e-2)
	parser.add_argument("--alpha", dest="alpha", type=np.float,
                    help="weighting of likelihoods (default is 0.5)", default=0.5)
	parser.add_argument("--true_communities", dest="true_communities", 
                help="path of csv file containing ground truth community memberships", default=None)

	args = parser.parse_args()

	return args

def main():

	args = parse_args()

	gml_file = args.gml_file
	gamma = args.gamma
	T = args.T
	stdout.write("Reading G from {} with scaling exponent={} and T={}\n".format(gml_file, gamma, T))
	N, R, A, order_of_appearance = preprocess_G(gml_file, gamma, T)
	stdout.write("Preprocessed G\n")

	# X, true_communities = generate_X(N, K, attribute_file, order_of_appearance)
	attribute_file = args.attribute_file
	stdout.write("Reading attributes from {}\n".format(attribute_file))
	X = preprocess_X(attribute_file)
	stdout.write("Preprocessed X\n")

	K = X.shape[1]
	stdout.write("K={}\n".format(K))

	C = args.num_communities
	stdout.write("C={}\n".format(C))

	thetas, M, W = initialize_matrices(N, C, K, R)
	stdout.write("Initialized matrices\n")

	eta = args.eta
	alpha = args.alpha
	lamb_F = args.lamb_F
	lamb_W = args.lamb_W
	num_epochs = args.num_epochs
	true_communities = preprocess_true_communities(args.true_communities)
	attribute_type = args.attribute_type

	stdout.write("Training with eta={}, num_epochs={}, lamb_F={}, lamb_W={}, alpha={}, attribute_type={}\n".format(eta,	
		num_epochs, lamb_F, lamb_W, alpha, attribute_type))

	thetas, M, W = train(A, X, N, K, C, R, thetas, M, W, 
		eta=eta, lamb_F=lamb_F, lamb_W=lamb_W, alpha=alpha, 
		num_epochs=num_epochs, true_communities=true_communities, attribute_type=attribute_type)

	stdout.write("Trained matrices\n") 

	# draw_network(N, C, R, thetas, M)
	
	# stdout.write("Visualised network\n")

	# thetas = pd.DataFrame(thetas)
	# M = pd.DataFrame(M)
	# W = pd.DataFrame(W)

	# thetas.to_csv("thetas.csv", sep=",", index=False, header=False)
	# M.to_csv("M.csv", sep=",", index=False, header=False)
	# W.to_csv("W.csv", sep=",", index=False, header=False)

	return

if __name__ == "__main__":
	main()