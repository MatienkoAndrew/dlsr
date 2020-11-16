import pandas as pd
from dslr.histogram_plot import histogram_plot


if __name__ == '__main__':
	df = pd.read_csv('datasets/dataset_train.csv')
	histogram_plot(df)