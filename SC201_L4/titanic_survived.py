"""
File: titanic_survived.py
Name:
----------------------------------
This file contains 3 of the most important steps
in machine learning:
1) Data pre-processing
2) Training
3) Predicting
"""
import math

TRAIN_DATA_PATH = 'titanic_data/train.csv'
NUM_EPOCHS = 1000
ALPHA = 0.01



def sigmoid(k):
    """
    :param k: float, linear function value
    :return: float, probability of the linear function value
    """
    return 1 / (1 + math.exp(-k))


def dot(lst1, lst2):
    """
    : param lst1: list, the feature vector
    : param lst2: list, the weights
    : return: float, the dot product of 2 list
    """
    return sum(lst1[i] * lst2[i] for i in range(len(lst1)))


def main():
    # Milestone 1
    training_data = data_pre_processing()
    print(training_data)

    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    weights = [0] * len(training_data[0][0])

    # Milestone 3
    training(training_data, weights)
    print()

    # Milestone 4
    predict(training_data, weights)


# Milestone 1
def data_pre_processing():
    """
    Read the training data from TRAIN_DATA_PATH and get ready for training!
    :return: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    """
    training_data = []
    is_head = True
    with open('titanic_data/train.csv', 'r') as f:
        for line in f:
            if is_head:
                is_head = False
            else:
                feature_vector, y = feature_extractor(line)
                training_data.append((feature_vector, y))

    return training_data


# Milestone 2
def feature_extractor(line):
    """
    : param line: str, the line of data extracted from the training set
    : return: Tuple(list, label), feature_vector and label of a passenger
    """
    line = line.strip()
    data_list = line.split(',')
    feature_vector = []
    label = int(data_list[1])
    for i in range(len(data_list)):
        if i == 2:
            # Pclass
            feature_vector.append((int(data_list[i]) - 1) / 2)
        elif i == 5:
            # Gender male=1, female=0
            if data_list[i] == 'male':
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        elif i == 6:
            age = data_list[i]
            # Age
            if age:
                feature_vector.append((float(age) - 0.42) / (80 - 0.42))
            else:
                feature_vector.append((29.699 - 0.42) / (80 - 0.42))
        elif i == 7:
            # SibSp
            feature_vector.append(int(data_list[i])/8)
        elif i == 8:
            # Parch
            feature_vector.append(int(data_list[i])/6)
        elif i == 10:
            # Fare
            feature_vector.append(float(data_list[i])/512.3292)
        elif i == 12:
            # Embarked, S=0, C=1, Q=2
            if data_list[i] == 'S':
                feature_vector.append(0)
            elif data_list[i] == 'C':
                feature_vector.append(1/2)
            else:
                feature_vector.append(2/2)
    return feature_vector, label


# Milestone 3
def training(training_data, weights):
    """
    : param training_data: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    : param weights: list[float], the weight vector (the parameters on each feature)
    """
    for epoch in range(NUM_EPOCHS):
        cost = 0
        for x, y in training_data:
            #################################
            k = dot(x,weights)
            h = sigmoid(k)
            loss = -(y*math.log(h)+(1-y)*math.log(1-h))
            cost += loss
            for j in range(len(weights)):
                weights[j] = weights[j] - ALPHA * (h-y)*x[j]

            # 控制Error message
            # raise Exception('Not implemented yet')

            #################################
        cost /= len(training_data)
        if epoch % 100 == 0:
            print('Cost over all data:', cost)


# Milestone 4
def predict(training_data, weights):
    """
    : param training_data: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    : param weights: list[float], the weight vector (the parameters on each feature)
    """
    acc = 0
    num_data = 0
    for x, y in training_data:
        predict = get_prediction(x, weights)
        print('True Label: ' + str(y) + ' --------> Predict: ' + str(predict))
        if y == predict:
            acc += 1
        num_data += 1
    print('---------------------------')
    print('Acc: ' + str(acc / num_data))


def get_prediction(x, weights):
    """
    : param x: list[float], the value of each data on
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    : param weights: list[float], the weight vector (the parameters on each feature)
    : return: float, the score of x (if it is > 0 then the passenger may survive)
    """
    k = dot(x, weights)
    if k > 0:
        return 1
    return 0
    # h = sigmoid(k)
    # if h > 0.5:
    #     return 1
    # return 0


if __name__ == '__main__':
    main()
