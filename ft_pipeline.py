from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CleanFeatures(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		X = X.drop(["condition", "Unnamed: 0"], axis=1)
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
		return X.to_numpy()
