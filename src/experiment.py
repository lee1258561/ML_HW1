import os
import sys
import numpy as np
from time import time

from sklearn.decomposition import PCA

from model import Model
from utils import *


def experiment1(data_dir='data', fig_dir='fig', scoring='f1', DTvisual=False):
	task = 'creditcard'
	filename = 'size-5000_porp-0.1.csv'
	fig_path = create_path(fig_dir, task, 'experiment')
	X_train, X_test, y_train, y_test = load_data(os.path.join(os.getcwd(), data_dir, task, filename), is_shuffle=True)

	exp_model = ['NeuralNet']

	print ("\t\tTrain Acc\tTest Acc\tVal Score\t\tTrue pos\tTrue neg\tFalse pos\tFlase neg\tF1 score\tTrain t\tTest t")
	for clf in exp_model:
		model = Model(clf=clf, 
					  scoring=scoring, 
					  model_suffix=task, 
					  load_params=True, 
					  verbose=False)
		print (model.model.get_params())
		model.generate_learning_curve(X_train, 
		                              y_train, 
		                              ylim=(0.0, 1.01), 
		                              train_sizes=np.linspace(.1, 1.0, 5),
		                              fig_path=fig_path,
		                              scoring='recall')
		model.generate_learning_curve(X_train, 
		                              y_train, 
		                              ylim=(0.0, 1.01), 
		                              train_sizes=np.linspace(.1, 1.0, 5),
		                              fig_path=fig_path,
		                              scoring='f1')
		start = time()
		train_acc, cv_scores, train_f1, report = model.train(X_train, y_train)
		train_end = time()
		y_pred, test_acc, c_matrix, f1, report = model.test(X_test, y_test=y_test)
		test_end = time()

		if clf == 'DecisionTree' and DTvisualize == True:
			from sklearn import tree
			import graphviz
			dot_data = tree.export_graphviz(model.model, out_file=None, rounded=True) 
			graph = graphviz.Source(dot_data) 
			graph.render(os.path.join(fig_path, 'DTVisualize'))

		print("%s\t%0.03f\t\t%0.03f\t\t%s\t%d\t\t%d\t\t%d\t\t%d\t\t%0.03f\t\t%0.03f\t\t%0.03f" %
			(clf + '\t' if len(clf) < 8 else clf,
			 train_acc, 
			 test_acc, 
			 "%0.3f (+/- %0.3f)" % (np.mean(cv_scores['test_score']), np.std(cv_scores['test_score'])),
			 c_matrix[1][1],
			 c_matrix[0][0],
			 c_matrix[0][1],
			 c_matrix[1][0],
			 f1,
			 train_end - start,
			 test_end - train_end))

	data size vs score
	f1_matrix = [[] for _ in range(len(exp_model))]
	precision_matrix = [[] for _ in range(len(exp_model))]
	recall_matrix = [[] for _ in range(len(exp_model))]
	prop_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
	for prop in prop_list:
		filename = 'size-5000_porp-%s.csv' % str(prop)
		X_train, X_test, y_train, y_test = load_data(os.path.join(os.getcwd(), data_dir, task, filename))

		for i in range(len(exp_model)):
			clf = exp_model[i]
			print ('Train %s for %s' % (clf, filename))
			model = Model(clf=clf, 
						  scoring=scoring, 
						  model_suffix=task, 
						  load_params=True, 
						  verbose=False)

			train_acc, cv_scores, train_f1, train_report = model.train(X_train, y_train, validate=False)
			y_pred, test_acc, c_matrix, f1, report = model.test(X_test, y_test=y_test)

			f1_matrix[i].append(f1)
			precision_matrix[i].append(report['1']['precision'])
			recall_matrix[i].append(report['1']['recall'])

	title = "F1_score"
	plot_and_save(np.array(prop_list),
				  np.array(f1_matrix),
				  exp_model,
				  title,
				  "Positive sample proportion",
				  "F1 score",
				  fig_path=os.path.join(fig_path, title))
	title = "Precision"
	plot_and_save(np.array(prop_list),
				  np.array(precision_matrix),
				  exp_model,
				  title,
				  "Positive sample proportion",
				  "Precision",
				  fig_path=os.path.join(fig_path, title))
	title = "Recall"
	plot_and_save(np.array(prop_list),
				  np.array(recall_matrix),
				  exp_model,
				  title,
				  "Positive sample proportion",
				  "Recall",
				  fig_path=os.path.join(fig_path, title))

	#Model Complexity vs score
	X_train, X_test, y_train, y_test = load_data(os.path.join(os.getcwd(), data_dir, task, filename))
	fig_labels = ['Training score', 'Validation score', 'Test score']
	DT_params = {
					'criterion': 'gini',
					'random_state': 123565432
				}
	max_depth = [1,2, 3, 5, 7, 9, 11, 15]
	DT_f1_score = [[] for _ in range(3)]
	for md in max_depth:
		DT_params['max_depth'] = md
		model = Model(clf='DecisionTree', 
					  scoring=scoring, 
					  model_suffix=task,  
					  verbose=False)
		train_acc, cv_scores, train_f1, train_report = model.train(X_train, y_train, params=DT_params, validate=True)
		y_pred, test_acc, c_matrix, f1, report = model.test(X_test, y_test=y_test)
		DT_f1_score[0].append(train_f1)
		DT_f1_score[1].append(np.mean(cv_scores['test_score']))
		DT_f1_score[2].append(f1)
		DT_params.pop('max_depth')

	ADABoost_params = {
						'learning_rate': 0.2,
						'random_state': 123565432
					  }
	n_estimators = [20, 50, 100, 200, 500, 1000, 2000, 5000]
	boost_f1_score = [[] for _ in range(3)]
	for ne in n_estimators:
		ADABoost_params['n_estimators'] = ne
		model = Model(clf='AdaBoost', 
					  scoring=scoring, 
					  model_suffix=task,  
					  verbose=False)
		train_acc, cv_scores, train_f1, train_report = model.train(X_train, y_train, params=ADABoost_params)
		y_pred, test_acc, c_matrix, f1, report = model.test(X_test, y_test=y_test)
		boost_f1_score[0].append(train_f1)
		boost_f1_score[1].append(np.mean(cv_scores['test_score']))
		boost_f1_score[2].append(f1)
		ADABoost_params.pop('n_estimators')

	title = "Decision Tree F1 Score"
	fig_path = create_path(fig_dir, task, filename=title)
	plot_and_save(np.array(max_depth),
				  np.array(DT_f1_score),
				  fig_labels,
				  title,
				  "Max tree depth",
				  "F1 score",
				  fig_path=fig_path)
	title = "AdaBoost F1 score"
	fig_path = create_path(fig_dir, task, filename=title)
	plot_and_save(np.array(n_estimators),
				  np.array(boost_f1_score),
				  fig_labels,
				  title,
				  "Number of estimators",
				  "F1 score",
				  fig_path=fig_path)
	
	model = Model(clf=clf, 
				  scoring=scoring, 
				  model_suffix=task, 
				  load_params=True, 
				  verbose=True)

	train_acc, cv_scores, train_f1, train_report = model.train(X_train, y_train, validate=False)


def experiment2(data_dir='data', fig_dir='fig', scoring='accuracy'):
	task = 'MNIST'
	filename = 'MNIST_4_9_size-1000.csv'
	file_path = os.path.join(os.getcwd(), data_dir, task, filename)
	count, total = analyze_data(file_path, threshold=1)
	X_train, X_test, y_train, y_test = load_data(file_path, is_shuffle=True)
	exp_model = ['SVM', 
				 'DecisionTree', 
				 'KNN', 
				 'AdaBoost', 
				 'NeuralNet']

	fig_path = create_path(fig_dir, task, 'experiment')
	print ("\t\tTrain Acc\tTest Acc\tVal Score\t\tTrue pos\tTrue neg\tFalse pos\tFlase neg\tF1 score\tTrain t\tTest t")
	for clf in exp_model:
		model = Model(clf=clf, 
					  scoring=scoring, 
					  model_suffix=task, 
					  load_params=True, 
					  verbose=False)
		# model.generate_learning_curve(X_train, 
		#                               y_train, 
		#                               ylim=(0.0, 1.01), 
		#                               train_sizes=np.linspace(.1, 1.0, 5),
		#                               fig_path=fig_path,
		#                               scoring='accuracy')
		start = time()
		train_acc, cv_scores, train_f1, report = model.train(X_train, y_train)
		train_end = time()
		y_pred, test_acc, c_matrix, f1, report = model.test(X_test, y_test=y_test)
		test_end = time()

		print("%s\t%0.03f\t\t%0.03f\t\t%s\t%d\t\t%d\t\t%d\t\t%d\t\t%0.03f\t\t%0.03f\t\t%0.03f" %
			(clf + '\t' if len(clf) < 8 else clf,
			 train_acc, 
			 test_acc, 
			 "%0.3f (+/- %0.3f)" % (np.mean(cv_scores['test_score']), np.std(cv_scores['test_score'])),
			 c_matrix[1][1],
			 c_matrix[0][0],
			 c_matrix[0][1],
			 c_matrix[1][0],
			 f1,
			 train_end - start,
			 test_end - train_end))



	fig_path = create_path(fig_dir, task, 'experiment', 'PCA')
	print ('Total number of features that is non zero: %d' % total)

	print ("Dimension reduction using PCA ...")
	pca = PCA(n_components=115)
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)
	print("Precentage of covarence preserved: %0.03f" % np.sum(pca.explained_variance_ratio_)) 

	print ("\t\tTrain Acc\tTest Acc\tVal Score\t\tTrue pos\tTrue neg\tFalse pos\tFlase neg\tF1 score\tTrain t\tTest t")
	for clf in exp_model:
		model = Model(clf=clf, 
					  scoring=scoring, 
					  model_suffix=task, 
					  load_params=True, 
					  verbose=False)
		model.generate_learning_curve(X_train, 
		                              y_train, 
		                              ylim=(0.0, 1.01), 
		                              train_sizes=np.linspace(.1, 1.0, 5),
		                              fig_path=fig_path,
		                              scoring='accuracy')
		start = time()
		train_acc, cv_scores, train_f1, report = model.train(X_train, y_train)
		train_end = time()
		y_pred, test_acc, c_matrix, f1, report = model.test(X_test, y_test=y_test)
		test_end = time()

		print("%s\t%0.03f\t\t%0.03f\t\t%s\t%d\t\t%d\t\t%d\t\t%d\t\t%0.03f\t\t%0.03f\t\t%0.03f" %
			(clf + '\t' if len(clf) < 8 else clf,
			 train_acc, 
			 test_acc, 
			 "%0.3f (+/- %0.3f)" % (np.mean(cv_scores['test_score']), np.std(cv_scores['test_score'])),
			 c_matrix[1][1],
			 c_matrix[0][0],
			 c_matrix[0][1],
			 c_matrix[1][0],
			 f1,
			 train_end - start,
			 test_end - train_end))

	dt_params = {
				'criterion': 'gini',
				'max_depth': 9,
				'random_state': 123565432
			}
	model = Model(clf='DecisionTree', 
				  scoring=scoring, 
				  model_suffix=task, 
				  verbose=True)
	train_acc, cv_scores, train_f1, report = model.train(X_train, y_train, params=dt_params)
	y_pred, test_acc, c_matrix, f1, report = model.test(X_test, y_test=y_test)

	print ("Train accuracy: %0.03f" % train_acc)
	print ("Test accuracy: %0.03f" % test_acc)

if __name__ == '__main__':
	if len(sys.argv) != 2 or sys.argv[1] not in ['creditcard', 'MNIST']:
		print ("Usage: python searchParameter.py {creditcard|MNIST}")
		sys.exit (1)
	if sys.argv[1] == 'creditcard':
		experiment1()
	else:
		experiment2()


