# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:37:15 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:37:17 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class LogisticRegression:
	def __init__(self, eta=0.1, max_iter=100, l2=0, initial_weights=None, multi_class=None):
		self.eta = eta
		self.max_iter = max_iter
		self.l2 = l2
		self._w = initial_weights
		self._K = multi_class
		self._errors = []
		self._loss = {}
		pass

	def sigmoid(self, prediction):
		return 1.0 / (1.0 + np.exp(-prediction))

	def fit(self, X, y, sample_weight=None):
		self._K = np.unique(y).tolist()
		X_bias = np.c_[(np.ones((len(X), 1))), X]
		m = X_bias.shape[0]

		self._w = sample_weight
		if not self._w:
			self._w = np.zeros(X_bias.shape[1] * len(self._K))
		self._w = self._w.reshape(len(self._K), X_bias.shape[1])

		yVec = np.zeros((len(y), len(self._K)))

		for i in range(len(y)):
			yVec[i, self._K.index(y[i])] = 1

		for i in range(self.max_iter + 1):
			y_pred = self.sigmoid(np.dot(self._w, X_bias.T))
			loss_function = (-1.0 / m) * (np.sum((yVec.T * np.log(y_pred) + (1 - yVec.T) * np.log(1 - y_pred))))

			gradients = -(1 / m) * np.dot((yVec - y_pred.T).T, X_bias)
			step_size = self.eta * gradients
			self._w = self._w - step_size
			if i % 10 == 0:
				self._loss[i] = loss_function
				pass
			pass
		return self

	def predict_proba(self, X):
		X_bias = np.c_[(np.ones((len(X), 1))), X]
		return self.sigmoid(np.dot(self._w, X_bias.T)).T

	def predict(self, X):
		probabilities = self.predict_proba(X)
		return [self._K[x] for x in probabilities.argmax(1)]
