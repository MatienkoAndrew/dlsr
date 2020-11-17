# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pair_plot.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:36:53 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:36:55 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
from dslr.pair_plot1 import pair_plot

if __name__ == '__main__':
	df = pd.read_csv('datasets/dataset_train.csv')
	pair_plot(df)
