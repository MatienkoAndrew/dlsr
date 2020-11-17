# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pair_plot1.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:37:52 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:37:53 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from matplotlib import pyplot as plt
import numpy as np

def pair_plot(df):
	courses = np.array(df.columns[6:])
	df3 = df[np.append('Hogwarts House', courses)]

	plt.figure(figsize=(35, 35))

	n_rows = 13
	n_cols = 13
	faculties = ['Hufflepuff', 'Ravenclaw', 'Gryffindor', 'Slytherin']

	for row, course in enumerate(courses):
		for col, course1 in enumerate(courses):
			index = row * n_cols + col
			ax = plt.subplot(13, 13, index + 1)
			for faculty in faculties:
				if row == col:
					ax.hist(df3[df3['Hogwarts House'] == faculty][course], alpha=0.7)
				else:
					ax.scatter(df3[df3['Hogwarts House'] == faculty][course1],
							   df3[df3['Hogwarts House'] == faculty][course])
					pass
				pass
			if ax.is_first_col():
				ax.set_ylabel(courses[row].replace(' ', '\n'))
			else:
				ax.tick_params(labelleft=False)

			if ax.is_last_row():
				ax.set_xlabel(courses[col].replace(' ', '\n'))
			else:
				ax.tick_params(labelbottom=False)

			pass
		pass

	plt.legend(faculties, loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))
	plt.show()
