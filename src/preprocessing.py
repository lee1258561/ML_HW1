import os
from random import shuffle
from sklearn.datasets import fetch_openml
from sklearn import datasets

from utils import *


def write_csv(data, filepath):
	with open(filepath, 'w') as f:
		for row in data:
			print(",".join(row), file=f)


def preprocess_creditcard(data_path, data_dir='data', size=None, proportion=0.01):
	raw_data = []
	with open(data_path, 'r') as f:
		for line in f:
			raw_data.append(line.strip().split(','))

	total_size = len(raw_data[1:])
	positive_data = []
	negetive_data = []
	for row in raw_data[1:]:
		row[-1] = row[-1][1]
		if row[-1] == '1': positive_data.append(row)
		else: negetive_data.append(row)

	shuffle(positive_data)
	shuffle(negetive_data)

	#processed = positive_data + negative_data 
	if not size: 
		size = total_size
	elif type(size) == type(int()): 
		assert size <= total_size, "Size must not be bigger than total size of data."
	elif type(size) == type(float()): 
		assert size > 0 and size <= 1, "Size proportion must be in the range of (0, 1]."
		size = int(size / float(total_size))
	else:
		raise ValueError

	pos_size = int(proportion * size)
	if pos_size > len(positive_data):
		print ("WARNING: Not enough positive data.")
		pos_size = len(positive_data)

	neg_size = size - pos_size
	if neg_size > len(negetive_data):
		print ("WARNING: Not enough negetive data.")
		neg_size = len(negetive_data)

	processed = positive_data[:pos_size] + negetive_data[:neg_size]

	filename = f"size-{size}_porp-{proportion}.csv"
	filepath = create_path(data_dir, 'creditcard', filename=filename)

	write_csv(processed, filepath)

def preprocess_MNIST(data_dir='data', size=None):
	MNIST_path = create_path(data_dir, filename='MNIST.csv')
	if not os.path.exists(MNIST_path):
		X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
		X, y = list(X), list(y)
		with open(MNIST_path, 'w') as f:
			for i in range(len(X)):
				print (','.join(map(str, X[i])) + ',' + str(y[i]), file=f)
	else:
		X, y = [], []
		with open(MNIST_path, 'r') as f:
			for line in f:
				instance = line.strip().split(',')
				X.append(instance[:-1])
				y.append(instance[-1])

	count = {}
	for label in y:
		if label not in count:
			count[label] = 0
		count[label] += 1

	print (count)

	label_to_X = {}
	for i in range(len(X)):
		feats, label = X[i], y[i]
		feats.append(label)
		if label not in label_to_X:
			label_to_X[label] = []

		label_to_X[label].append(feats)

	pos_label = '4'
	neg_label = '9'

	assert size <= len(label_to_X[pos_label]), "Size greater than number of instances for label %s" % pos_label
	assert size <= len(label_to_X[neg_label]), "Size greater than number of instances for label %s" % neg_label 

	shuffle(label_to_X[pos_label])
	shuffle(label_to_X[neg_label])
	for i in range(size):
		label_to_X[pos_label][i][-1] = '1'
		label_to_X[neg_label][i][-1] = '0'

	processed = label_to_X[pos_label][:size] + label_to_X[neg_label][:size]
	filename = f'MNIST_{pos_label}_{neg_label}_size-{size}.csv'
	filepath = create_path(data_dir, 'MNIST', filename=filename)

	write_csv(processed, filepath)

if __name__ == "__main__":
	#preprocess_creditcard('./data/creditcard.csv', size=5000, proportion=0.1)
	preprocess_MNIST(size=1000)

    
