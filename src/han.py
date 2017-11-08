import numpy as np

def update_theta_u(u, N, C, A, X, R, thetas, M, W, eta, alpha, lamb_F):
	
	# compute F
	delta_theta = np.pi - abs(np.pi - abs(thetas[:,None] - M[:,1]))
	x = np.cosh(R[:,None]) * np.cosh(M[:,0]) - np.sinh(R[:,None]) * np.sinh(M[:,0]) * np.cos(delta_theta)
	h = np.arccosh(x)
	F = np.exp(- h ** 2 / (2 * M[:,2] ** 2))
	P = 1 - np.exp(-F.dot(F.T))
	
	# partial delta theta partial theta
	partial_delta_theta_u_partial_theta_u = np.array([-np.sign(np.pi - abs(thetas[u] - M[c,1])) *\
	 -np.sign(thetas[u] - M[c,1]) * 1 for c in range(C)])
	
	# parital h partial theta
	partial_x_u_partial_delta_theta_u = np.diag([np.sinh(R[u]) * np.sinh(M[c,0]) * np.sin(delta_theta[u,c])
		for c in range(C)])

	partial_h_u_partial_x_u = np.diag([1 / np.sqrt(x[u, c] ** 2 - 1) for c in range(C)])
	 
	# partial F partial theta
	partial_F_u_partial_h_u = np.diag([-h[u, c] / M[c,2] ** 2 * np.exp(-h[u, c] ** 2 / (2 * M[c,2] ** 2))
		for c in range(C)])

	partial_P_u_partial_F_u = np.array([[F[v, c] * np.exp(-F[u].dot(F[v])) for c in range(C)]
		for v in range(N)])
	
	# partial L_G
	partial_L_G_u_partial_P_u = np.array([A[u, v] / P[u, v] - (1 - A[u, v]) / (1 - P[u, v])
		for v in range(N)])

	# print partial_delta_theta_u_partial_theta_u.shape
	# print partial_x_u_partial_delta_theta_u.shape
	# print partial_h_u_partial_x_u.shape
	print partial_F_u_partial_h_u.shape
	print partial_F_u_partial_h_u
	# print partial_P_u_partial_F_u.shape
	# print partial_L_G_u_partial_P_u.shape
	
	# print "start"
	partial_L_G_u_partial_theta_u = partial_L_G_u_partial_P_u.dot(partial_P_u_partial_F_u).dot(partial_F_u_partial_h_u)
	# print type(partial_L_G_u_partial_theta_u)
	# print "end"
	# print partial_L_G_u_partial_theta_u.shape, partial_h_u_partial_x_u.shape
	# print partial_L_G_u_partial_theta_u
	# print partial_L_G_u_partial_theta_u.shape
	# print partial_h_u_partial_x_u
	partial_L_G_u_partial_theta_u.dot(partial_h_u_partial_x_u)
	# .dot(partial_x_u_partial_delta_theta_u)\
	# .dot(partial_delta_theta_u_partial_theta_u)

	# partial L_X
	# F_u = np.append(F[u], 1)
	# Q_u = compute_Q_u(F_u, W)
	# partial_L_X_u_partial_F_u = np.array([(X[u] - Q_u).dot(W[:,c]) for c in range(C)])
	# partial_L_X_u_partial_theta_u = partial_L_X_u_partial_F_u.dot(partial_F_u_partial_theta_u)
	
	# return thetas[u] + eta * ((1 - alpha) * partial_L_G_u_partial_theta_u)

def main():

	u = 0
	N = 500
	C = 10
	K = 10
	A = np.random.randint(2, size=(N, N))
	X = np.random.randint(2, size=(N, K))
	R = np.random.rand(N)
	thetas = np.random.rand(N)
	M = np.random.rand(C, 3)
	W = np.random.rand(K, C+1)
	eta = 1e-2
	alpha=0
	lamb_F=0


	update_theta_u(u, N, C, A, X, R, thetas, M, W, eta, alpha, lamb_F)

	return

if __name__ == "__main__":
	main()