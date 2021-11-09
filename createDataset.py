import numpy as np
import csv
import matplotlib.pyplot as plt

with open('./dataset/mnist_train.csv', 'r') as infile, open('./dataset/mnist_train_38.csv', 'w') as outfile:
	writer = csv.writer(outfile)
	for row in csv.reader(infile):
		if row[0] == "3" or row[0] == "8":
			writer.writerow(row)

