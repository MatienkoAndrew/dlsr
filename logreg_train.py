# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_train.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:36:47 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:36:49 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import argparse
from dslr.train_test_split import train_test_split
from dslr.scaler import StandardScaler
from dslr.model import LogisticRegression
from dslr.stochastic_model import StochasticLogisticRegression
from dslr.minibatch_logreg import MiniBatchLogisticRegression
import numpy as np

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
	parser.add_argument('-w', action="store_true", help="Models' weights")
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
	y_test = y_test.reset_index(drop=True)
	logreg = LogisticRegression(max_iter=1000)
	logreg.fit(X_train_scaled, y_train)

	##-- Stochastic Logistic Regression
	slogreg = StochasticLogisticRegression(max_iter=1000)
	slogreg.fit(X_train_scaled, y_train)

	##-- Mini Batch Logistic Regression
	minibatch = MiniBatchLogisticRegression(max_iter=1000)
	minibatch.fit(X_train_scaled, y_train)

	if (args.s):
		plt.plot(logreg._loss.keys(), logreg._loss.values(), marker='o', c='red')
		plt.plot(slogreg._loss.keys(), slogreg._loss.values(), marker='o', c='blue')
		plt.plot(minibatch._loss.keys(), minibatch._loss.values(), marker='o', c='green')
		plt.xlabel('step')
		plt.ylabel('Loss')
		plt.grid(True)
		plt.legend(['LogisticRegression', 'StochasticLogisticRegression',
					'Mini Batch Logistic Regression'])
		plt.show()


	if (args.loss):
		plt.plot(logreg._loss.keys(), logreg._loss.values(), marker='o')
		plt.xlabel('step')
		plt.ylabel('Loss')
		plt.grid(True)
		plt.show()

	if (args.k):
		print("LogisticRegression:")
		print("\tFalse(Train):", sum([y_train[i] != logreg.predict(X_train_scaled)[i]
										for i, _ in enumerate(y_train)]), "of", len(y_train))
		print("\tFalse(Test):", sum([y_test[i] != logreg.predict(X_test_scaled)[i]
							   for i, _ in enumerate(y_test)]), "of", len(y_test))
		print("\tAccuracy score(Train):", accuracy_score(y_train, logreg.predict(X_train_scaled)))
		print("\tAccuracy score(Test):", accuracy_score(y_test, logreg.predict(X_test_scaled)))
		print("Stochastic LR")
		print("\tFalse(Train):", sum([y_train[i] != slogreg.predict(X_train_scaled)[i]
										for i, _ in enumerate(y_train)]), "of", len(y_train))
		print("\tFalse(Test):", sum([y_test[i] != slogreg.predict(X_test_scaled)[i]
							   for i, _ in enumerate(y_test)]), "of", len(y_test))
		print("\tAccuracy score(Train):", accuracy_score(y_train, slogreg.predict(X_train_scaled)))
		print("\tAccuracy score(Test):", accuracy_score(y_test, slogreg.predict(X_test_scaled)))
		print("Mini Batch LR")
		print("\tFalse(Train):", sum([y_train[i] != minibatch.predict(X_train_scaled)[i]
										for i, _ in enumerate(y_train)]), "of", len(y_train))
		print("\tFalse(Test):", sum([y_test[i] != minibatch.predict(X_test_scaled)[i]
							   for i, _ in enumerate(y_test)]), "of", len(y_test))
		print("\tAccuracy score(Train):", accuracy_score(y_train, minibatch.predict(X_train_scaled)))
		print("\tAccuracy score(Test):", accuracy_score(y_test, minibatch.predict(X_test_scaled)))

	if args.w:
		print("Logistic Regression:")
		print(pd.DataFrame(logreg._w, columns=np.append("Intercept", X_train.columns)))
		print("Stochastic Logistic Regression:")
		print(pd.DataFrame(slogreg._w, columns=np.append("Intercept", X_train.columns)))
		print("Mini Batch Logistic Regression:")
		print(pd.DataFrame(minibatch._w, columns=np.append("Intercept", X_train.columns)))

	pd.DataFrame(logreg._w).to_csv('weights.csv', index=False)
