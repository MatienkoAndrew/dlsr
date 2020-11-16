from matplotlib import pyplot as plt
import numpy as np

def histogram_plot(train):
	plt.figure(1, figsize=(10, 10))
	faculties = ['Hufflepuff', 'Ravenclaw', 'Gryffindor', 'Slytherin']
	courses = np.array(train.columns[6:])

	n_rows = 13
	n_cols = 4
	for row, course in enumerate(courses):
		for col, faculty in enumerate(faculties):
			index = row*n_cols + col
			plt.subplot(13, 4, index + 1)
			plt.hist(train[train['Hogwarts House'] == faculty][course].dropna())
			plt.title(faculty)
			plt.ylabel(course)
	plt.subplots_adjust(wspace=0.5, hspace=2)
	#plt.show()

	plt.figure(2, figsize=(20, 2))
	faculties = ['Hufflepuff', 'Ravenclaw', 'Gryffindor', 'Slytherin']
	for i, faculty in enumerate(faculties):
		plt.subplot(1, 4, i + 1)
		plt.hist(train[train['Hogwarts House'] == faculty]['Care of Magical Creatures'])
		plt.title(faculty)
		plt.xlabel('Number of students')
		plt.ylabel('Marks')
	#plt.show()

	plt.figure(3, figsize=(10, 6))
	faculties = ['Hufflepuff', 'Ravenclaw', 'Gryffindor', 'Slytherin']
	for faculty in faculties:
		plt.hist(train[train['Hogwarts House'] == faculty]['Care of Magical Creatures'], alpha=0.7)
		plt.title('Care of Magical Creatures')
		plt.legend(faculties)
		plt.xlabel('Number of students')
		plt.ylabel('Marks')
	plt.show()