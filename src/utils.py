from sklearn.model_selection import train_test_split
from sklearn import datasets
from random import shuffle

import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")

def plot_learning_curve(train_scores_mean,
                        train_scores_std,
                        val_scores_mean,
                        val_scores_std,
                        train_sizes,
                        ylim=None,
                        title='test',
                        fig_path='fig',
                        format='png'):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid(True, linestyle = "-.", color = '0.3')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(fig_path + '/' + title + '.' + format, format=format)
    plt.clf()

def plot_and_save(x, ys, labels, title, x_axis, y_axis, axis_range=None, ylim=None, fig_path='fig', format='png'):
    if axis_range is None:
        plt.axis([x[0], x[-1], 0, 1])
    elif type(axis_range) == type(list()):
        plt.axis(axis_range)
    elif axis_range == 'auto':
        pass

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    lines = []
    for y, label in zip(ys, labels):
        l, = plt.plot(x, y, 'o-', label=label)
    plt.legend(loc="best")
    plt.grid(True, linestyle = "-.", color = '0.3')

    plt.savefig(fig_path + '.' + format, format=format)
    plt.clf()

def create_path(*arg, filename=None):
    path = os.getcwd()
    for directory in arg:
        path = os.path.join(path, directory)
        if not os.path.exists(path):
            print('%s doesn\'t exist, creating...' % path)
            os.mkdir(path)

    if filename:
        path = os.path.join(path, filename)
    return path


def print_score(scores, scoring, train=False):
    if type(scoring) != type([]):
        if train:
            print("Train: %0.2f (+/- %0.2f)" % (np.mean(scores['train_score']), np.std(scores['train_score']) * 2))

        print("Cross validation: %0.2f (+/- %0.2f)" % (np.mean(scores['test_score']), np.std(scores['test_score']) * 2))
        return

    for s_method in scoring:
        if train:
            print("Train: %0.2f (+/- %0.2f)" % (np.mean(scores['train_' + s_method]), np.std(scores['train_' + s_method]) * 2))

        print("Cross validation: %0.2f (+/- %0.2f)" % (np.mean(scores['test_' + s_method]), np.std(scores['test_' + s_method]) * 2))

def unit_test_data():
    digits = datasets.load_digits()

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    print(X.shape) 
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    return X_train, X_test, y_train, y_test

def load_data(data_path, split_prop=0.2, is_shuffle=False):
    pos_X, neg_X = [], []
    with open(data_path, 'r') as f:
        for line in f:
            instance = list(map(float, line.strip().split(',')))
            if instance[-1] == 1.0:
                pos_X.append(instance[:-1])
            else:
                neg_X.append(instance[:-1])

    pos_test_size, neg_test_size = int(split_prop * len(pos_X)), int(split_prop * len(neg_X))
    pos_train_size, neg_train_size = len(pos_X) - pos_test_size, len(neg_X) - neg_test_size
    
    X_test, y_test = pos_X[:pos_test_size] + neg_X[:neg_test_size], [1] * pos_test_size + [0] * neg_test_size
    X_train, y_train = pos_X[pos_test_size:] + neg_X[neg_test_size:], [1] * pos_train_size + [0] * neg_train_size

    assert len(X_train) == len(y_train) and len(X_test) == len(y_test), "Dimention of X and y must be the same."

    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    if is_shuffle:
        train_indices = list(range(X_train.shape[0]))
        shuffle(train_indices)
        test_indices = list(range(X_test.shape[0]))
        shuffle(test_indices)
        X_train, X_test, y_train, y_test = X_train[train_indices], X_test[test_indices], y_train[train_indices], y_test[test_indices]

    return X_train, X_test, y_train, y_test

def load_gisette(data_dir='./data/GISETTE/'):
    X_train_path = os.path.join(data_dir, 'gisette_train.data')
    X_train = []
    with open(X_train_path, 'r') as f:
        for line in f:
            X_train.append(list(map(int, line.strip().split(' '))))
    X_train = np.array(X_train)

    X_test_path = os.path.join(data_dir, 'gisette_valid.data')
    X_test = []
    with open(X_test_path, 'r') as f:
        for line in f:
            X_test.append(list(map(int, line.strip().split(' '))))
    X_test = np.array(X_test)

    y_train_path = os.path.join(data_dir, 'gisette_train.labels')
    y_train = []
    with open(y_train_path, 'r') as f:
        for line in f:
            y_train.append(int(line.strip()))
    y_train = np.array(y_train)
    y_train[y_train == -1] = 0

    y_test_path = os.path.join(data_dir, 'gisette_valid.labels')
    y_test = []
    with open(y_test_path, 'r') as f:
        for line in f:
            y_test.append(int(line.strip()))
    y_test = np.array(y_test)
    y_test[y_test == -1] = 0

    assert X_train.shape[0] == y_train.shape[0] and X_test.shape[0] == y_test.shape[0], "Dimention of X and y must be the same."
    return X_train, X_test, y_train, y_test


def analyze_data(data_path, threshold=50):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            instance = list(map(float, line.strip().split(',')))
            data.append(instance)

    count = [0] * len(data[0])
    for instance in data:
        for i in range(len(instance)):
            if instance[i] != 0.0:
                count[i] += 1

    total = 0
    for c in count:
        if c >= threshold:
            total += 1


    return count, total


    