import numpy as np

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