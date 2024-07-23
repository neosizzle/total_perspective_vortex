
import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
import numpy as np

from sklearn import datasets, decomposition

class FtPca:
	def __init__(self, n_components = 2):
		self.n_components = n_components

	def fit(self, data):
		# generate covariance matrix
		cov_matrix = np.cov(data, rowvar=False)

		# calculate eigenvector and eigenvalue pairs for the covarience matrix above
		eigens = np.linalg.eig(cov_matrix)

		# sort and keep components based on n_componenets
		e_values = eigens.eigenvalues
		e_vectors = eigens.eigenvectors
		e_val_map = []
		for idx, val in enumerate(e_values):
			e_val_map.append({
				"og_idx": idx,
				"val": val
			})
		sorted_e_values = sorted(e_val_map, key=lambda x: x['val'], reverse=True)
		kept_e_values = list(map(lambda x: x['og_idx'], sorted_e_values[:self.n_components]))
		kept_e_vectors = e_vectors[:, kept_e_values]

		# project
		self.feature_matrix = kept_e_vectors
	
	def transform(self, data):
		return np.dot(self.feature_matrix.T , data.T ).T
	
np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plt.tight_layout(pad=3)

ft_pca = FtPca(n_components=2)
ft_pca.fit(X)
ft_X = ft_pca.transform(X)

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
# print(ft_X)
print("===========")
# print(X)

ax.scatter(X.T[0], X.T[1], c=y)

plt.savefig("test.png")