0. git clone https://github.com/lee1258561/ML_HW1.git
1. Setup
	using python 3.7.2
	run:
		pip install sklearn, matplotlib

3. Usage (run under root dir):
	python python src/searchParameter.py {creditcard|MNIST} 
		Grid search parameter for five algorithms in given dataset, and generate learning curve for best parameter. The search space can be change by editing grid variables in src/searchParameter.py

	python src/searchParameter.py {creditcard|MNIST} {SVM|DecisionTree|KNN|AdaBoost|NeuralNet}
		Generate learning curve for given algorithm on given dataset. The parameter of the dataset can be change by editing params variables in src/searchParameter.py

	python src/experiment.py {creditcard|MNIST}
		Run experiment of given classification problem and generate chart for reports


