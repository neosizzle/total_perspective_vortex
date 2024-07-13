from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
target = data.target # need to generate this?
frame = data.frame

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
frame = scaler.fit_transform(frame) 


from sklearn.model_selection import train_test_split
data_train, target_train, data_test, target_test = train_test_split(frame, target, test_size=0.3, random_state=1)

print(data_train)

# print("Accuracy:", metrics.accuracy_score(target_test, pred))
