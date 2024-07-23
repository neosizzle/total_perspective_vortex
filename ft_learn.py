import argparse
import coloredlogs, logging
import pandas as pd
import time
import numpy as np
import warnings
import ft_pipeline

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neural_network  import MLPClassifier

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_pipeline(verbose=True):

	parameter_grid = {
		"alpha": (0.1, 0.2, 0.3, 0.4, 0.5),
		"momentum": (0.1, 0.2, 0.3, 0.4, 0.5)
	}

	# this uses cross_val_score under the hood
	classifier = RandomizedSearchCV(
		estimator=MLPClassifier(max_iter=1000, random_state=69, verbose=verbose),
		param_distributions=parameter_grid,
		random_state=42,
		n_jobs=4,
		n_iter=5,
		verbose=4,
	)
	return Pipeline([
		('cleaner', ft_pipeline.CleanFeatures()),
		('scaler', ft_pipeline.FtStandardScaler()),
		# ('scaler', StandardScaler()),
		('decomposer', ft_pipeline.FtPca(n_components=40)),
		# ('decomposer', PCA(n_components=40)),
		('classifier', classifier)
	])

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('-n', '--nostreaming', help="Disable streaming of test data, use whole test_data when predict all at once", action='store_true')
	parser.add_argument('-d', '--data', help="Path of the csv data file for training / predicting", type=str, default='data.csv')
	return parser.parse_args()

def main():
	coloredlogs.install()
	warnings.filterwarnings('ignore')

	args = get_args()
	verbose = args.verbose

	frame = pd.read_csv('data.csv')
	target = pd.Series(frame['condition'].values)

	treatment_pipeline = create_pipeline(verbose)
	train_data, test_data, train_target, test_target = train_test_split(frame, target, test_size=0.4, random_state=1)

	logging.info("Fitting pipeline..")
	treatment_pipeline.fit(train_data, train_target, scaler__use_prev_weights=False)
	logging.info("pipeline fit OK")

	chunks_size = 100
	data_chunks =  np.array_split(test_data, chunks_size)
	target_chunks = np.array_split(test_target, chunks_size)

	# prediction
	if args.nostreaming:
		score = treatment_pipeline.score(test_data, test_target, scaler__use_prev_weights=True)
		logging.info(f"Accuracy: {score}")
		return
	for chunk_data, chunk_target in zip(data_chunks, target_chunks):
		prediction = treatment_pipeline.predict(chunk_data)
		if verbose:
			logging.info(f"chunk_data {chunk_data} pred {prediction}")
		logging.info(f"Accuracy for current chunk {accuracy_score(chunk_target, prediction)}")
		time.sleep(2)


main()