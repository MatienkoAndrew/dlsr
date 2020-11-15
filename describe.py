import pandas as pd
import numpy as np
import argparse

def per_fun(df_col, p):
	if df_col.shape == (0,):
		return np.nan
	df_sort = np.sort(df_col)
	k = (len(df_sort) - 1) * (p / 100)
	f = np.floor(k)
	c = np.ceil(k)

	if f == c:
		return df_sort[int(k)]

	d0 = df_sort[int(f)] * (c - k)
	d1 = df_sort[int(c)] * (k - f)
	return d0 +d1

def max_fun(df_col):
	if df_col.shape == (0,):
		return np.nan
	max_v = -100000000.0
	for x in df_col:
		if x > max_v:
			max_v = x
	return max_v

def min_fun(df_col):
	if df_col.shape == (0,):
		return np.nan
	min_v = 100000000.0
	for x in df_col:
		if x < min_v:
			min_v = x
	return min_v


def std_val(df_col, mean):
	if df_col.shape == (0,):
		return np.nan
	sum = 0.0
	for x in df_col:
		sum += (x - mean) ** 2
	return (sum / (len(df_col) - 1)) ** 0.5

def mean_val(df_col):
	if df_col.shape == (0,):
		return np.nan
	sum = 0.0
	for x in df_col:
		sum += x
	return sum / len(df_col)

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