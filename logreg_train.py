import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import argparse
from dslr.train_test_split import train_test_split
from dslr.scaler import StandardScaler
from dslr.model import LogisticRegression
from dslr.stochastic_model import StochasticLogisticRegression

# print(pd.DataFrame(logreg._w))
# print(logreg.predict_proba(X_train_scaled[:10]))
# print(logreg.predict(X_train_scaled[:10]))
# print(np.array(y_train[:10]).reshape(1, -1))

#pd.DataFrame(logreg._w).to_csv('weights.csv', index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="input dataset")
	parser.add_argument('-l', '--loss', action="store_true", help="Show model")
	parser.add_argument('-k', action="store_true", help="Loss function")
	parser.add_argument('-s', action="store_true", help="Compare models")
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

	slogreg = StochasticLogisticRegression()
	slogreg.fit(X_train_scaled, y_train)

	if (args.s):
		plt.plot(logreg._loss.keys(), logreg._loss.values(), marker='o', c='red')
		plt.plot(slogreg._loss.keys(), slogreg._loss.values(), marker='o', c='blue')
		plt.xlabel('step')
		plt.ylabel('Loss')
		plt.grid(True)
		plt.legend(['LogisticRegression', 'StochasticLogisticRegression'])
		plt.show()


	if (args.loss):
		plt.plot(logreg._loss.keys(), logreg._loss.values(), marker='o')
		plt.xlabel('step')
		plt.ylabel('Loss')
		plt.grid(True)
		plt.show()

	if (args.k):
		print("LogisticRegression:")
		print("\tAccuracy score(Train):", accuracy_score(y_train, logreg.predict(X_train_scaled)))
		print("\tAccuracy score(Test):", accuracy_score(y_test, logreg.predict(X_test_scaled)))
		print("Stochastic LR")
		print("\tAccuracy score(Train):", accuracy_score(y_train, slogreg.predict(X_train_scaled)))
		print("\tAccuracy score(Test):", accuracy_score(y_test, slogreg.predict(X_test_scaled)))

	pd.DataFrame(logreg._w).to_csv('weights.csv', index=False)
