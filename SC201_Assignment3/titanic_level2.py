"""
File: titanic_level2.py
Name:
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle. Hyperparameters are hidden by the library!
This abstraction makes it easy to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
    """
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'
			 data, if the mode is 'Test'
	"""
    data = pd.read_csv(filename)
    labels = None
    data['Sex'].replace('male', 1, inplace=True)
    data['Sex'].replace('female', 0, inplace=True)
    data['Embarked'].replace('Q', 2, inplace=True)
    data['Embarked'].replace('S', 0, inplace=True)
    data['Embarked'].replace('C', 1, inplace=True)
    data[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']].round(0)
    data['Age'].round(1)
    data['Fare'].round(3)
    ################################
    #                              #
    #             TODO:            #
    #                              #
    ################################
    if mode == 'Train':
        data = data.loc[:, ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        data.dropna(inplace=True)
        labels = data['Survived']
        data = data.drop(columns=['Survived'])
        return data, labels
    elif mode == 'Test':
        data = data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        columns = data.columns
        for column in columns:
            if data[column].isna().any() == True:
                data[column].fillna(value=training_data[column].mean().round(3), inplace=True)
        return data


def one_hot_encoding(data, feature):
    """
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
    ############################
    #                          #
    #          TODO:           #
    #                          #
    ############################
    one_hot_data = pd.get_dummies(data[feature])
    columns = one_hot_data.columns
    for i in range(len(columns)):
        one_hot_data.rename(columns={columns[i] : feature+'_'+str(i)},inplace= True)
    data = data.drop(columns=feature)
    data = data.join(one_hot_data)
    return data


def standardization(data, mode='Train'):
    """
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
    ############################
    #                          #
    #          TODO:           #
    #                          #
    ############################
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    return data


def main():
    """
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy
	on degree1; ~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimals)
	TODO: real accuracy on degree1 -> ______________________
	TODO: real accuracy on degree2 -> ______________________
	TODO: real accuracy on degree3 -> ______________________
	"""
    train_data = data_preprocess(TRAIN_FILE, mode='Train')
    test_data = data_preprocess(TEST_FILE, mode='Test', training_data=train_data)
    train_data = one_hot_encoding(train_data, ('Sex','Pclass','Embarked'))



if __name__ == '__main__':
    main()
