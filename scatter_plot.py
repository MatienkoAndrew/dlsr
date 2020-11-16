import pandas as pd
from dslr.scatter_plot1 import scatter_plot

if __name__ == '__main__':
	df = pd.read_csv('datasets/dataset_train.csv')
	scatter_plot(df)