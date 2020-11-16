import pandas as pd
from dslr.pair_plot1 import pair_plot

if __name__ == '__main__':
	df = pd.read_csv('datasets/dataset_train.csv')
	pair_plot(df)