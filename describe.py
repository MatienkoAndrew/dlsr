# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    describe.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:34:57 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:35:08 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import argparse
from dslr.math1 import *

def describe(filename):
	df = pd.read_csv(filename)
	count = []
	mean = []
	std = []
	min_val = []
	per1 = []
	per2 = []
	per3 = []
	max_val = []

	i = -1
	for col in df.columns:
		if df[col].dtypes != 'object':
			i += 1
			df_temp = df[col][~np.isnan(df[col])]
			count.append(len(df_temp))
			mean.append(mean_val(df_temp))
			std.append(std_val(df_temp, mean[i]))
			min_val.append(min_fun(df_temp))
			per1.append(per_fun(df_temp, 25))
			per2.append(per_fun(df_temp, 50))
			per3.append(per_fun(df_temp, 75))
			max_val.append(max_fun(df_temp))

	columns = [col for col in df.columns if df[col].dtypes != 'object']
	index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
	data = [count, mean, std, min_val, per1, per2, per3, max_val]
	des = pd.DataFrame(data, columns=columns, index=index)
	return des

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("datasets", type=str, help="input dataset")
	args = parser.parse_args()
	print(describe(args.datasets))
