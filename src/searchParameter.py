from model import Model
from utils import *

import os
import sys

import warnings
warnings.simplefilter("ignore")

random_state = 123565432

#===== Searching Grid of five algorithm =====
SVM_grids = [{	
				'kernel': ['rbf'], 
				'gamma': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
				'C': [0.1, 1, 10, 100, 1000, 10000, 100000],
				'max_iter': [100000],
				'random_state': [random_state]
			 },
			 {
			 	'kernel': ['poly'], 
			 	'degree': [1, 2, 3, 4, 5, 6],
				'gamma': [1e-5, 1e-6, 1e-7],
				'C': [0.0001, 0.001, 100000, 1000000],
				'max_iter': [100000],
				'random_state': [random_state]
			 },
			 {
			 	'kernel': ['linear'], 
			 	'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
			 	'max_iter': [100000],
			 	'random_state': [random_state]
			 }]

DT_grids = [{
				'criterion': ['gini', 'entropy'],
				'max_depth':[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
				#'class_weight': [None, {1: 10, 0: 1}],
				'random_state': [random_state]
			}]
KNN_grids = [{
				'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 20, 50, 100, 200],
				'weights': ['uniform', 'distance'],
				'p': [1, 2], # L1 or L2 distance
			 }]
ADABoost_grids = [{
					'n_estimators': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
					'learning_rate': [0.1, 0.2, 0.5, 0.8],
					'random_state': [random_state]
				  }]
NN_grids = [{
				'solver':['adam'], 
				'alpha': [1e-4, 1e-5, 1e-6], 
				'hidden_layer_sizes': [(64), (100), (200), (32, 32), (64, 32, 16), (64, 64, 32, 16)], 
				'random_state': [random_state],
				'activation': ['logistic', 'tanh', 'relu'],
				'learning_rate': ['adaptive'],
				'learning_rate_init': [1e-4, 1e-5, 1e-6],
				'max_iter': [1000, 2000],
			 }]

model_grid_map = {
					'SVM': SVM_grids,
					'DecisionTree': DT_grids,
					'KNN': KNN_grids,
					'AdaBoost': ADABoost_grids,
					'NeuralNet': NN_grids
				 }
# =============================================

# ======== Indivisual test parameters =========
#nn_params = {'activation': 'logistic', 'alpha': 1e-06, 'hidden_layer_sizes': (64, 32, 16), 'learning_rate': 'adaptive', 'learning_rate_init': 0.0001, 'max_iter': 5000, 'random_state': 123565432, 'solver': 'adam'}
nn_params = {
				"activation": "relu", 
				"alpha": 1e-04, 
				"hidden_layer_sizes": (200), 
				"learning_rate": "adaptive", 
				"learning_rate_init": 1e-05, 
				"max_iter": 1000, 
				"random_state": 123565432, 
				"solver": "adam"}
svm_params = {
			 	'kernel':'rbf', 
			 	'C': 1000000,
			 	#'degree': 5,
			 	'gamma': 1e-7,
			 	#'coef0': 0.1,
			 	'max_iter': 1000000,
			 	#'tol': 1e-4,
			 	'random_state': random_state
			 }
dt_params = {
				'criterion': 'gini',
				'max_depth': 2,
				'random_state': random_state
			}

ada_params = {
				'n_estimators': 2000,
				'learning_rate': 0.1,
			 }

knn_params = {
				'n_neighbors': 3,
				'weights': 'uniform',
				'p': 1, # L1 or L2 distance
			 }
params_map = {
				'SVM': svm_params,
				'DecisionTree': dt_params,
				'KNN': knn_params,
				'AdaBoost': ada_params,
				'NeuralNet': nn_params
			 }

def search(filename, task, data_dir='data', scoring='accuracy', fold_num=5, fig_dir='fig', test=False, f1_fig=False):
	if test:
		X_train, X_test, y_train, y_test = unit_test_data()
	else:
		if task == 'GISETTE':
			file_path = os.path.join(os.getcwd(), data_dir, task)
			X_train, X_test, y_train, y_test = load_gisette(data_dir=file_path)
		elif task == 'creditcard' or task == 'MNIST':
			file_path = os.path.join(os.getcwd(), data_dir, task, filename)
			X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
		elif task == 'MNIST':
			file_path = os.path.join(os.getcwd(), data_dir, task, filename)
			X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)

	model_list = ['SVM', 'DecisionTree', 'KNN', 'AdaBoost', 'NeuralNet']

	fig_path = create_path(fig_dir, task, 'search')
	for m_method in model_list:
		model = Model(clf=m_method, scoring=scoring, model_suffix=task, fold_num=fold_num)
		best_params = model.search_hyperparameter(X_train, 
												  y_train, 
												  model_grid_map[m_method], 
												  X_test=X_test, 
												  y_test=y_test)
		model.generate_learning_curve(X_train, 
		                              y_train, 
		                              ylim=(0.0, 1.01), 
		                              train_sizes=np.linspace(.1, 1.0, 5),
		                              fig_path=fig_path,
		                              params=best_params,
		                              scoring='accuracy')
		if f1_fig:
			model.generate_learning_curve(X_train, 
			                              y_train, 
			                              ylim=(0.0, 1.01), 
			                              train_sizes=np.linspace(.1, 1.0, 5),
			                              fig_path=fig_path,
			                              params=best_params,
			                              scoring='f1')
	
def run_one(filename, task, clf, data_dir='data', scoring='accuracy', fold_num=5, fig_dir='fig', test=False, f1_fig=False):
	if test:
		X_train, X_test, y_train, y_test = unit_test_data()
	else:
		if task == 'GISETTE':
			file_path = os.path.join(os.getcwd(), data_dir, task)
			X_train, X_test, y_train, y_test = load_gisette(data_dir=file_path)
		elif task == 'creditcard':
			file_path = os.path.join(os.getcwd(), data_dir, task, filename)
			X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
		elif task == 'MNIST':
			file_path = os.path.join(os.getcwd(), data_dir, task, filename)
			X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
	print (X_train.shape)
	model = Model(clf=clf, 
				  scoring='accuracy', 
				  model_suffix='test',  
				  verbose=False)

	fig_path = create_path(fig_dir, task, 'test')
	if clf == 'DecisionTree':
		from sklearn import tree
		import graphviz
		model.train(X_train, y_train, params=params_map[clf])
		dot_data = tree.export_graphviz(model.model, out_file=None, rounded=True) 
		graph = graphviz.Source(dot_data) 
		graph.render("test")

	if f1_fig:
		model.generate_learning_curve(X_train, 
		                              y_train, 
		                              ylim=(0.0, 1.01), 
		                              train_sizes=np.linspace(.1, 1.0, 5),
		                              fig_path=fig_path,
		                              params=params_map[clf],
		                              scoring='f1')
		model.generate_learning_curve(X_train, 
		                              y_train, 
		                              ylim=(0.0, 1.01), 
		                              train_sizes=np.linspace(.1, 1.0, 5),
		                              fig_path=fig_path,
		                              params=params_map[clf],
		                              scoring='recall')
	else:
		model.generate_learning_curve(X_train, 
	                              y_train, 
	                              ylim=(0.8, 1.01), 
	                              train_sizes=np.linspace(.1, 1.0, 5),
	                              fig_path=fig_path,
	                              params=params_map[clf],
	                              scoring='accuracy')

if __name__ == '__main__':
	if len(sys.argv) < 2 or len(sys.argv) > 3 :
		print ("Usage: python searchParameter.py {creditcard|MNIST} [{SVM|DecisionTree|KNN|AdaBoost|NeuralNet]")
		sys.exit (1)

	if sys.argv[1] not in ['creditcard', 'MNIST'] or (len(sys.argv) == 3 and sys.argv[2] not in ['SVM', 'DecisionTree', 'KNN', 'AdaBoost', 'NeuralNet']):
		print ("Usage: python searchParameter.py {creditcard|MNIST} [{SVM|DecisionTree|KNN|AdaBoost|NeuralNet]")
		sys.exit (1)

	if sys.argv[1] == 'creditcard':
		if len(sys.argv) == 2:
			search('size-5000_porp-0.1.csv', 'creditcard', scoring='f1', fold_num=5, f1_fig=True)
		else:
			run_one('size-5000_porp-0.1.csv', 'creditcard', sys.argv[2], scoring='f1', fold_num=5, f1_fig=True)
	elif sys.argv[1] == 'MNIST':
		if len(sys.argv) == 2:
			search('MNIST_4_9_size-1000.csv', 'MNIST')
		else:
			run_one('MNIST_4_9_size-1000.csv', 'MNIST', sys.argv[2])
	






