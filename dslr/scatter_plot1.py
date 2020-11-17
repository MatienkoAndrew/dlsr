# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    scatter_plot1.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 16:37:59 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 16:38:01 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from matplotlib import pyplot as plt

def scatter_plot(df):
	df2 = df[['Hogwarts House', 'Astronomy', 'Defense Against the Dark Arts']].dropna()
	plt.figure(1, figsize=(12, 6))
	plt.scatter(df2['Astronomy'], df2['Defense Against the Dark Arts'])
	plt.xlabel('Astronomy')
	plt.ylabel('Defense Against the Dark Arts')
	#plt.show()

	plt.figure(2, figsize=(12, 6))
	faculties = ['Hufflepuff', 'Ravenclaw', 'Gryffindor', 'Slytherin']
	faculties = ['Hufflepuff', 'Ravenclaw', 'Gryffindor', 'Slytherin']
	for faculty in faculties:
		plt.scatter(df2[df2['Hogwarts House'] == faculty]['Astronomy'],
					df2[df2['Hogwarts House'] == faculty]['Defense Against the Dark Arts'])
	plt.legend(faculties)
	plt.xlabel('Astronomy')
	plt.ylabel('Defense Against the Dark Arts')
	plt.legend(faculties)
	plt.xlabel('Astronomy')
	plt.ylabel('Defense Against the Dark Arts')
	plt.show()
