import pandas as pd
from matplotlib import pyplot as plt

# print(pd.DataFrame(logreg._w))
# print(logreg.predict_proba(X_train_scaled[:10]))
# print(logreg.predict(X_train_scaled[:10]))
# print(np.array(y_train[:10]).reshape(1, -1))

#pd.DataFrame(logreg._w).to_csv('weights.csv', index=False)

import argparse
from dslr.train_test_split import train_test_split
from dslr.scaler import StandardScaler
from dslr.model import LogisticRegression

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="input dataset")
	parser.add_argument('-l', '--loss', action="store_true", help="Show model")
	args = parser.parse_args()

	train = pd.read_csv(args.dataset)
	df4 = train.fillna(method='ffill')

	X = df4.iloc[:, 6:].copy()
	y = df4['Hogwarts House'].copy()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.fit_transform(X_test)

	y_train = y_train.reset_index(drop=True)
	logreg = LogisticRegression()
	logreg.fit(X_train_scaled, y_train)

	if (args.loss):
		plt.plot(logreg._loss.keys(), logreg._loss.values(), marker='o')
		plt.xlabel('step')
		plt.ylabel('Loss')
		plt.grid(True)
		plt.show()

	pd.DataFrame(logreg._w).to_csv('weights.csv', index=False)
