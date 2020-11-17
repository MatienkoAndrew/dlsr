# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_test_split.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:38:05 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:38:07 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def train_test_split(X, y, test_size=0.3, shuffle=True, random_state=None):
	np.random.seed(random_state)

	n = X.shape[0]

	if shuffle == True:
		shuffle_array_index = np.random.permutation(range(len(X)))
		X = X.iloc[shuffle_array_index]
		y = y.iloc[shuffle_array_index]

		train_size = int(((1 - test_size) * 100) * n / 100)
		X_train = X[:train_size]
		X_test = X[train_size:]
		y_train = y[:train_size]
		y_test = y[train_size:]
		return X_train, X_test, y_train, y_test
	else:
		train_size = int(((1 - test_size) * 100) * n / 100)
		X_train = X[:train_size]
		X_test = X[train_size:]
		y_train = y[:train_size]
		y_test = y[train_size:]
		return X_train, X_test, y_train, y_test
