# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    histogram.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:35:22 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:35:33 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
from dslr.histogram_plot import histogram_plot

if __name__ == '__main__':
	df = pd.read_csv('datasets/dataset_train.csv')
	histogram_plot(df)
