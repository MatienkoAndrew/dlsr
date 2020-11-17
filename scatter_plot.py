# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    scatter_plot.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:36:32 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:36:33 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
from dslr.scatter_plot1 import scatter_plot

if __name__ == '__main__':
	df = pd.read_csv('datasets/dataset_train.csv')
	scatter_plot(df)
