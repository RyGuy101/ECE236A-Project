import numpy as np
import csv
import matplotlib.pyplot as plt
import random

num_train = 1000

m_1 = [-1,1]
m_2 = [1,-1]

with open('./dataset/gaussian_train.csv', 'w') as outfile:
	writer = csv.writer(outfile)

	label = np.random.randint(2,size=num_train).transpose()
	u1 = []
	u2 = []
	for l in label:
		if l == 0:
			u1.append(np.random.normal(m_1[0],1))
			u2.append(np.random.normal(m_1[1],1))
		elif l == 1:
			u1.append(np.random.normal(m_2[0],1))
			u2.append(np.random.normal(m_2[1],1))
	u1 = np.array(u1)
	u2 = np.array(u2)
	dataset = np.stack((label,u1,u2), axis=-1)
	for d in dataset:
		writer.writerow(d.tolist())

	plt.figure(1)
	plt.scatter(u1,u2,c=label)
	plt.show()

	#if row[0] == "3" or row[0] == "8":
	#	writer.writerow(row)

