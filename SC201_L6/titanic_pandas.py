"""
File: titanic_pandas.py
Name: 
---------------------------
This file shows how to pandas and sklearn
packages to build a machine learning project
from scratch by their high order abstraction.
The steps of this project are:
1) Data pre-processing by pandas
2) Learning by sklearn
3) Test on D_test
"""

import pandas as pd
from sklearn import linear_model, preprocessing


# Constants - filenames for data set
TRAIN_FILE = 'titanic_data/train.csv'             # Training set filename
TEST_FILE = 'titanic_data/test.csv'               # Test set filename

# Global variable
nan_cache = {}                                    # Cache for test set missing data


def main():

	# Data cleaning
	train_data = data_preprocess(TRAIN_FILE, mode='Train')
	test_data = data_preprocess(TEST_FILE, mode='Test')
	train_data = one_hot_encoding(train_data)
	test_data = one_hot_encoding(test_data)
	# Extract true labels
	Y = train_data.pop('Survived')

	# Abandon features ('PassengerId', 'Name', 'Ticket', 'Cabin')

		##方法1. pop
	train_data.pop('PassengerId')
	train_data.pop('Name')
	train_data.pop('Ticket')
	train_data.pop('Cabin')

		##方法2.要的保留
	# features = ['Pclass','Age','Sex','SibSp','Parch','Embarked']
	# train_data =train_data[features]
	# print(train_data.count)


	# Extract features ('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')

	# Normalization / Standardization
	normalizer = preprocessing.MinMaxScaler()
	X = normalizer.fit_transform(train_data)

	#############################
	# Degree 1 Polynomial Model #
	#############################
	h = linear_model.LogisticRegression()  #還有什麼模型？？？
	classifier = h.fit(X, Y)
	train_acc = classifier.score(X, Y)
	print('Training ACC:', train_acc)

	# Test
	test_data.pop('PassengerId')
	test_data.pop('Name')
	test_data.pop('Ticket')
	test_data.pop('Cabin')
		## Fit-Transform or transform實際上做什麼？
	X_test = normalizer.transform(test_data)
	predict = classifier.predict(X_test)
	print(predict)
	out_file(predict, 'pandas_sklearn_dregree1.csv')

	#############################
	# Degree 2 Polynomial Model #
	#############################
	h = linear_model.LogisticRegression()
	poly_phi = preprocessing.PolynomialFeatures(degree=3)
	X_poly = poly_phi.fit_transform(X)
	classifier_poly = h.fit(X_poly, Y)
	train_acc = classifier_poly.score(X_poly,Y)
	print('Training ACC: ', train_acc)
	# Test dataset
	X_test_poly = poly_phi.transform(X_test)
	predict_poly = classifier_poly.predict(X_test_poly)
	out_file(predict_poly, 'pandas_sklearn_degree3.csv')
	

def data_preprocess(filename, mode='Train'):
	"""
	: param filename: str, the csv file to be read into by pd
	: param mode: str, the indicator of training mode or testing mode
	-----------------------------------------------
	This function reads in data by pd, changing string data to float, 
	and finally tackling missing data showing as NaN on pandas
	"""

	# Read in data as a column based DataFrame
	data = pd.read_csv(filename)
	print(data.count())
	if mode == 'Train':
		# Cleaning the missing data in Age column by replacing NaN with its median
		age_median = data.Age.median()
		# Bad way ##data['Age'].fillna(age_median)
		data['Age'].fillna(age_median , inplace=True)
		# print(data.Age)
		# Filling the missing data in Embarked column with 'S'
		data['Embarked'].fillna('S', inplace= True)

		# Cache some data for test set (Age and Fare)
		nan_cache['Age'] = age_median
		nan_cache['Fare'] = data.Fare.median()

	else:
		# Fill in the NaN cells by the values in nan_cache to make it consistent
			##將Train data的中位數加進Test data_set中
		data['Age'].fillna(nan_cache['Age'], inplace= True)
		data['Fare'].fillna(nan_cache['Fare'], inplace= True)

	# Changing 'male' to 1, 'female' to 0
	data.loc[data.Sex == 'male' , 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0

	# Changing 'S' to 0, 'C' to 1, 'Q' to 2
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2

	return data
	

def out_file(predictions, filename):
	"""
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	"""
	print('\n===============================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('PassengerId,Survived\n')
		start_id = 892
		for ans in predictions:
			out.write(str(start_id)+','+str(ans)+'\n')
			start_id += 1
	print('===============================================')


def one_hot_encoding(data):
	"""
	:param data: pd.DataFrame, the 2D data
	------------------------------------------------
	Extract important categorical data, making it a new one-hot vector
	"""
	# One hot encoding for a new category Male
	data['Male'] = 0
	data.loc[data.Sex ==1 , 'Male'] = 1
	# One hot encoding for a new category Female
	data['Female'] = 0
	data.loc[data.Sex == 0, 'Female'] = 1
	# No need Sex anymore!
	data.pop('Sex')
	# One hot encoding for a new category FirstClass
	data['FisrtClass'] = 0
	data.loc[data.Pclass == 1, 'FisrtClass'] = 1
	# One hot encoding for a new category SecondClass
	data['SecondClass'] = 0
	data.loc[data.Pclass == 2, 'SecondClass'] = 1
	# One hot encoding for a new category ThirdClass
	data['ThirdClass'] = 0
	data.loc[data.Pclass == 3, 'ThirdClass'] = 1
	# No need Pclass anymore!
	data.pop('Pclass')
	return data


if __name__ == '__main__':
	main()
