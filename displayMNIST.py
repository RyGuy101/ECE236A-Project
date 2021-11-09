import numpy as np
import csv
import matplotlib.pyplot as plt
from random import randrange

filename = './dataset/test/mnist_test_38.csv'
index = randrange(100)

with open(filename, newline='') as csvfile:
	reader = csv.reader(csvfile)
	row = next(reader)
	for i in range(index):
		row = next(reader)
	
	label = row[0]
	print(label)

	data = np.array(row[1:]).astype(np.int16).reshape(28,28)

	fig, ax = plt.subplots()
	ax.pcolormesh(np.flipud(data))
	ax.set_aspect(1)
	plt.show()
