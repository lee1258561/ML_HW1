0. git clone https://github.com/lee1258561/ML_HW1.git
	The data and model parameters in this repository is sufficient to reproduce the result. Download the original data only if you want to run the data preprocessing script.
1. Setup
	using python 3.7.2
	run:
		pip install sklearn, matplotlib

	Download the data and place it under data/ if you want to run preporcessing.py:
		credit card fraud detection: https://www.kaggle.com/mlg-ulb/creditcardfraud.
		MNIST data will be downloaded automatically by runing the preprocessing.py.

3. Usage (run under root dir):
	python src/preprocessing.py
		prepropess credit card fraud data and MNIST data, download the data first if you want to run it.
	python src/searchParameter.py {creditcard|MNIST} 
		Grid search parameter for five algorithms in given dataset, and generate learning curve for best parameter. The search space can be change by editing grid variables in src/searchParameter.py.

	python src/searchParameter.py {creditcard|MNIST} {SVM|DecisionTree|KNN|AdaBoost|NeuralNet}
		Generate learning curve for given algorithm on given dataset. The parameter of the dataset can be change by editing params variables in src/searchParameter.py.

	python src/experiment.py {creditcard|MNIST}
		Run experiment of given classification problem and generates chart for report.


