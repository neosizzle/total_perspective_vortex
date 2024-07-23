from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CleanFeatures(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		X = X.drop(["condition", "Unnamed: 0", "epoch"], axis=1)
		return X


class FtStandardScaler(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.mean = None
		self.stddev = None

	def fit(self, X, y=None, use_prev_weights=False):
		if use_prev_weights :
			if self.mean is None:
				raise ValueError("There is no previous weights to use and use_prev_weights is True")
			return self
		self.mean = X.mean()
		self.stddev = X.std()
		return self
	
	def transform(self, X):
		X = (X -self.mean)/self.stddev
		res = X.to_numpy()
		# print(res.shape)
		return res

class FtPca(BaseEstimator, TransformerMixin):
	def __init__(self, n_components = 5):
		self.n_components = n_components

	def fit(self, data, y=None):
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
		return self
	
	def transform(self, data):
		return np.dot(self.feature_matrix.T , data.T ).T