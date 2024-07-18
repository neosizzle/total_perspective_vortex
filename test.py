from sklearn.datasets import load_iris
import pandas as pd

frame = pd.read_csv('data.csv')
target = pd.Series(frame['condition'].values)
frame = frame.drop(["condition", "Unnamed: 0"], axis=1)
print(frame.info())

# data = load_iris(as_frame=True)
# target = data.target # need to generate this?
# frame = data.frame
# print(target)

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network  import MLPClassifier
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
	('classifier', MLPClassifier(max_iter=1000, random_state=69, verbose=True))
])

from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(frame, target, test_size=0.4, random_state=1)

pipeline.fit(train_data, train_target)
print("Accuracy:", pipeline.score(test_data, test_target))
