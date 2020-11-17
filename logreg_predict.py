# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_predict.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:36:09 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:36:13 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from dslr.model import LogisticRegression
from dslr.scaler import StandardScaler
import argparse
import pandas as pd

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="input_dataset")
	parser.add_argument("weights", type=str, help="input weights")
	args = parser.parse_args()

	test = pd.read_csv(args.dataset)
	df4 = test.fillna(method='ffill')

	X_test = df4.iloc[:, 6:].copy()

	scaler = StandardScaler()
	X_test_scaled = scaler.fit_transform(X_test)

	logreg = LogisticRegression(initial_weights=pd.read_csv(args.weights),
								multi_class=['Ravenclaw', 'Hufflepuff', 'Gryffindor', 'Slytherin'])
	y_pred = logreg.predict(X_test_scaled)

	k = pd.DataFrame(y_pred)
	pd.DataFrame(y_pred, columns=['Hogwarts House']).to_csv('houses.csv', index_label='Index')
