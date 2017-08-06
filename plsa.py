import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def normalize(ndarray, axis):
	return ndarray / ndarray.sum(axis = axis, keepdims = True)
	
def normalized_random_array(d0, d1):
	ndarray = np.random.rand(d0, d1)
	return normalize(ndarray, axis = 1)

if __name__ == "__main__":
	
	# initialize parameters
	D, K, V = 1000, 3, 6
	theta = normalized_random_array(D, K)
	phi = normalized_random_array(K, V)
	theta_est = normalized_random_array(D, K)
	phi_est = normalized_random_array(K, V)

	# for generate documents
	_theta = np.array([theta[:, :k+1].sum(axis = 1) for k in range(K)]).T
	_phi = np.array([phi[:, :v+1].sum(axis = 1) for v in range(V)]).T

	# generate documents
	W, Z = [], []
	N = np.random.randint(100, 300, D)
	for (d, N_d) in enumerate(N):
		Z.append((np.random.rand(N_d, 1) < _theta[d, :]).argmax(axis = 1))
		W.append((np.random.rand(N_d, 1) < _phi[Z[-1], :]).argmax(axis = 1))
	
	# estimate parameters
	q = np.zeros((D, np.max(N), K))
	T = 300
	likelihood = np.zeros(T)
	for t in range(T):
		print(t)

		# E step
		for (d, N_d) in enumerate(N):
			q[d, :N_d, :] = normalize(theta_est[d, :] * phi_est[:, W[d]].T, axis = 1)

		# M step
		theta_est[:, :] = normalize(q.sum(axis = 1), axis = 1)
		q_sum = np.zeros((K, V))
		for (d, W_d) in enumerate(W):
			v, index, count = np.unique(W_d, return_index= True, return_counts = True)
			q_sum[:, v[:]] += count[:] * q[d, index[:], :].T
		phi_est[:, :] = normalize(q_sum, axis = 1)

		# likelihood
		for (d, W_d) in enumerate(W):
			likelihood[t] += np.log(theta_est[d, :].dot(phi_est[:, W_d])).sum()

	# perplexity
	perplexity = np.exp(-likelihood[:] / N.sum())

	# print
	print('theta')
	print(theta)
	print('theta_est')
	print(theta_est)
	print('phi')
	print(phi)
	print('phi_est')
	print(phi_est)

	# plot
	plt.plot(np.arange(T), likelihood)
	plt.xlabel('iterations')
	plt.ylabel('likelihood')
	plt.ticklabel_format(scilimits = (0, 0), axis ='y')
	plt.savefig('likelihood.png')
	plt.clf()
	plt.plot(np.arange(T), perplexity)
	plt.xlabel('iterations')
	plt.ylabel('perplexity')
	plt.savefig('perplexity.png')
	plt.clf()
