import os 
import json
import random

from time import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.externals import joblib

import numpy as np

from utils import *
import warnings
warnings.simplefilter("ignore")

class Model():
    def __init__(self, 
                 params={}, 
                 clf='SVM', 
                 fold_num=5,
                 scoring='accuracy',
                 model_dir='model',
                 model_suffix='1',
                 data_dir='data', 
                 pre_trained=False, 
                 load_params=False,
                 n_jobs=4,
                 verbose=True):
        self.model_dir = model_dir
        self.model_suffix = model_suffix
        self.data_dir = data_dir
        self.preTrained = pre_trained
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.clf = clf
        self.params = params
        self.fold_num = fold_num
        self.scoring = scoring

        #Form path to model
        self.path = create_path(self.model_dir, self.clf)
        self.cv = StratifiedKFold(n_splits=self.fold_num, shuffle=False, random_state=None)

        if load_params:
            self.load_best_params()

        if not pre_trained:
            if self.clf == 'SVM': self.model = SVC(**self.params)
            elif self.clf == 'DecisionTree': self.model = DecisionTreeClassifier(**self.params)
            elif self.clf == 'KNN': self.model = KNeighborsClassifier(**self.params)
            elif self.clf == 'AdaBoost': self.model = AdaBoostClassifier(**self.params)
            elif self.clf == 'NeuralNet': self.model = MLPClassifier(**self.params)
            else: raise ValueError('Invalid clf name: %s' % self.clf)
        else:
            #load model
            self.load_model()

    def load_best_params(self, path=None):
        if not path:
            param_path = os.path.join(self.path, ('params_%s.json' % self.model_suffix))
        else: 
            param_path = path
        assert (os.path.exists(param_path)), ('%s doesn\'t exist...' % param_path)

        with open(param_path, 'r') as f:
            self.params = json.load(f)

    def load_model(self, path=None):
        if not path:
            model_path = os.path.join(self.path, ('model_%s.pkl' % self.model_suffix))
        else:
            model_path = path
        assert (os.path.exists(model_path)), ('%s doesn\'t exist...' % model_path)

        self.model = joblib.load(model_path)

    def search_hyperparameter(self, X_train, y_train, param_grids, X_test=None, y_test=None):
        #cv = StratifiedKFold(n_splits=self.fold_num, shuffle=False, random_state=None)
        gs = GridSearchCV(self.model, param_grids, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs)
        gs.fit(X_train, y_train)

        #Dump best parameters
        param_path = os.path.join(self.path, ('params_%s.json' % self.model_suffix))
        with open(param_path, 'w') as f:
            json.dump(gs.best_params_, f)
        
        if self.verbose:
            means = gs.cv_results_['mean_test_score']
            stds = gs.cv_results_['std_test_score']
            print("# Tuning hyper-parameters for %s, %s, %s" % (self.clf, self.model_suffix, self.scoring))
            print()
            print("Best parameters set found on development set:")
            print()
            print(gs.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            for mean, std, params in zip(means, stds, gs.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()
        train_acc, cv_scores, train_f1, report = self.train(X_train, y_train, params=gs.best_params_)
        if X_test is not None and y_test is not None:
            test_stats = self.test(X_test, y_test)

        return gs.best_params_

    def generate_learning_curve(self, 
                                X, y, 
                                ylim=None, 
                                params=None, 
                                scoring='auto',
                                train_sizes=np.linspace(.1, 1.0, 5),
                                fig_path='fig'):
        if scoring == 'auto':
            scoring = self.scoring

        if params is not None:
            self.model.set_params(**params)
            self.params = params


        train_sizes, train_scores, test_scores = learning_curve(self.model, 
                                                                X, y, 
                                                                cv=self.cv, 
                                                                scoring=scoring,
                                                                n_jobs=self.n_jobs, 
                                                                train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(test_scores, axis=1)
        val_scores_std = np.std(test_scores, axis=1)


        title = self.clf + '_' + scoring
        plot_learning_curve(train_scores_mean,
                            train_scores_std,
                            val_scores_mean,
                            val_scores_std,
                            train_sizes,
                            ylim=ylim,
                            title=title,
                            fig_path=fig_path)


    def train(self, X_train, y_train, params=None, validate=True):
        if params is not None:
            self.model.set_params(**params)
            self.params = params

        if self.verbose:
            print ("Train %s, using following parameters:" % self.clf)
            print (self.model.get_params())
     
        self.model.fit(X_train, y_train)


        #train accurency
        y_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred)
        train_f1 = f1_score(y_train, y_pred)
        report = classification_report(y_train, y_pred, output_dict=True) 
        
        #dump model to model dir
        model_path = os.path.join(self.path, ('model_%s.pkl' % self.model_suffix))
        joblib.dump(self.model, model_path)

        if validate:
            cv_scores = cross_validate(self.model, X_train, y_train, scoring=self.scoring, cv=self.cv, return_train_score=False)
            if self.verbose:
                print_score(cv_scores, self.scoring)
        else:

            cv_scores = {}


        if self.verbose: 
            print('Training accuracy: %0.3f' % train_acc)
            print('Model saved at: ./%s/' % model_path)

        return train_acc, cv_scores, train_f1, report

    #
    def test(self, X_test, y_test=None):
        y_pred = self.model.predict(X_test)

        if y_test is not None:
            test_acc = accuracy_score(y_test, y_pred)
            c_matrix = confusion_matrix(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            if self.verbose:
                print("Detailed classification report:")
                print()
                print(classification_report(y_test, y_pred))
                print()
                print("Testing accuracy: %0.3f" % test_acc)
                print()
                print("Confusion matrix:")
                print(c_matrix)
                print()
                print("F1 score: %0.3f" % f1)

            return y_pred, test_acc, c_matrix, f1, report
        else:
            return y_pred
        
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = unit_test_data()

    # # Set the parameters by cross-validation
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    # model = Model(clf='SVM')
    # best_params = model.search_hyperparameter(X_train, y_train, tuned_parameters, X_test=X_test, y_test=y_test)
    # train_score, scores = model.train(X_train, y_train, params=best_params)
    # y_pred = model.test(X_test, y_test=y_test)

    # model2 = Model(clf='SVM', preTrained=True)
    # model2.test(X_test, y_test=y_test)

    # print (os.getcwd())
    # with open('./model/SVM/params_1.json', 'r') as f:
    #     params = json.load(f)

    # model3 = Model(clf='SVM')
    # model3.train(X_train, y_train, params=params)
    # X_train, X_test, y_train, y_test = load_data('./data/creditcard/size-5000_porp-0.1.csv')
    # model = Model(clf='DecisionTree', 
    #               scoring='f1', 
    #               model_suffix='creditcard', 
    #               load_params=True, 
    #               verbose=False)

    # train_acc, cv_scores, train_f1, report = model.train(X_train, y_train)
    # y_pred, test_acc, c_matrix, f1, test_report = model.test(X_test, y_test=y_test)
    # print (train_f1)
    # print (report)
    # print (f1)
    # print (test_report)
    model = AdaBoostClassifier(verbose=True)
    model.fit(X_train, y_train)



