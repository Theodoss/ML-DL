"""
File: titanic_deep_nn.py
Name: 
-----------------------------
This file demonstrates how to create a deep
neural network (5 layers NN) to train our
titanic data. Your code should use all the
constants and global variables.
You should see the following Acc if you
correctly implement the deep neural network
Acc: 0.8431372549019608
-----------------------------
X.shape = (N0, m)
Y.shape = (1, m)
W1.shape -> (N0, N1)
W2.shape -> (N1, N2)
W3.shape -> (N2, N3)
W4.shape -> (N3, N4)
W5.shape -> (N4, N5)
B1.shape -> (N1, 1)
B2.shape -> (N2, 1)
B3.shape -> (N3, 1)
B4.shape -> (N4, 1)
B5.shape -> (N5, 1)
"""

from collections import defaultdict
import numpy as np

# Constants
TRAIN = 'titanic_data/train.csv'  # This is the filename of interest
NUM_EPOCHS = 40000  # This constant controls the total number of epochs 40000
ALPHA = 0.01  # This constant controls the learning rate α
L = 5  # This number controls the number of layers in NN
NODES = {  # This Dict[str: int] controls the number of nodes in each layer
    'N0': 6,
    'N1': 5,
    'N2': 4,
    'N3': 3,
    'N4': 2,
    'N5': 1
}


def main():
    """
    Print out the final accuracy of your deep neural network!
    You should see 0.8431372549019608
    """
    X_train, Y = data_preprocessing()
    _, m = X_train.shape
    print('Y.shape', Y.shape)
    print('X.shape', X_train.shape)
    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = normalize(X_train)
    ####################################
    #                                  #
    #              TODO:               #
    #                                  #
    ####################################
    weights, biases = neural_network(X, Y)
    A = X
    for i in range(1, len(NODES)-1):
        K = weights['W' + str(i)].T.dot(A) + biases['B' + str(i)]
        # print(f"K_Shape{i}{K_dict['K'+str(i)].shape}= W{i}{weights['W'+str(i)].shape}.T.Dot A{i}{A_dict['A'+str(i-1)].shape}+B{i} {biases['B'+str(i)].shape} ")
        A = np.maximum(0, K)

    # Finally Node
    K_Final = weights['W' + str(L)].T.dot(A) + biases['B' + str(L)]
    # print(f"K_Final{node_num}{K_Final.shape}= W{node_num}{weights['W' + str(node_num)].shape}.T.Dot A{node_num}{A_dict['A' + str(node_num - 1)].shape}+B{node_num} {biases['B' + str(node_num)].shape} ")
    scores = K_Final  # Already has prediction power
    predictions = np.where(scores > 0, 1, 0)
    acc = np.equal(predictions, Y)
    print(f'Acc = {np.sum(acc)/m}')



def normalize(X):
    """
    :param X: numpy_array, the dimension is (num_phi, m)
    :return: numpy_array, the values are normalized, where the dimension is still (num_phi, m)
    """
    min_array = np.min(X, axis=1, keepdims=True)
    max_array = np.max(X, axis=1, keepdims=True)
    return (X - min_array) / (max_array - min_array)


def neural_network(X, Y):
    """
    :param X: numpy_array, the array holding all the training data
    :param Y: numpy_array, the array holding all the ture labels in X
    :return (weights, bias): the tuple of parameters of this deep NN
             weights: Dict[str, float], key is 'W1', 'W2', ...
                                        value is the corresponding float
             bias: Dict[str, float], key is 'B1', 'B2', ...
                                     value is the corresponding float
    """
    np.random.seed(1)
    weights = {}
    biases = {}
    K_dict = {}
    A_dict = {}

    # Initialize all the weights and biases
    for i in range(1,len(NODES)):
        weights['W' + str(i)] = np.random.rand(NODES['N' + str(i-1)], NODES['N' + str(i)]) - 0.5
        biases['B' + str(i)] = np.random.rand(NODES['N' + str(i)], 1) - 0.5

    print(weights.keys(), biases.keys())

    #####################################
    #                                   #
    #               TODO:               #
    #                                   #
    #####################################
    node_num = len(NODES)-1
    n, m = X.shape
    print_every = 500
    for epoch in range(NUM_EPOCHS):

        # Forward Pass
        # TODO:
        A_dict['A0'] = X
        # Loop the first four nodes!
        for i in range(1,node_num):
            K_dict['K'+str(i)] = weights['W'+str(i)].T.dot(A_dict['A'+str(i-1)]) + biases['B'+str(i)]
            # print(f"K_Shape{i}{K_dict['K'+str(i)].shape}= W{i}{weights['W'+str(i)].shape}.T.Dot A{i}{A_dict['A'+str(i-1)].shape}+B{i} {biases['B'+str(i)].shape} ")
            A_dict['A'+str(i)] = np.maximum(0, K_dict['K'+str(i)])

        # Final Node
        K_Final = weights['W'+str(node_num)].T.dot(A_dict['A'+str(node_num-1)]) + biases['B'+str(node_num)]
        # print(f"K_Final{node_num}{K_Final.shape}= W{node_num}{weights['W' + str(node_num)].shape}.T.Dot A{node_num}{A_dict['A' + str(node_num - 1)].shape}+B{node_num} {biases['B' + str(node_num)].shape} ")
        scores = K_Final
        # Sigmoid
        H = 1 / (1 + np.exp(-scores))
        J = (1 / m) * np.sum(-(Y * np.log(H) + (1-Y) * np.log(1-H)))
        if epoch % print_every == 0:
            print(f'Cost in {epoch} epoch: {J} ')

        #Prepare dWi, dBi
        dW_dict = {}
        dB_dict = {}

        # Backward Pass
        # TODO:
        dK = (1/m)*np.sum(H-Y, axis=0, keepdims= True)
        dW_dict['dW'+str(node_num)] = np.dot(A_dict['A'+str(node_num-1)], dK.T)
        dB_dict['dB'+str(node_num)] = np.sum(dK, axis= 1, keepdims= True)
        # print(A_dict.keys())

        for i in range(node_num-1,0,-1):
            # print(i,weights['W'+str(i+1)].shape, dK.shape )
            dA = np.dot(weights['W'+str(i+1)], dK)
            dK = dA*np.where(K_dict['K'+str(i)] > 0, 1, 0)
            dW_dict['dW'+str(i)] = np.dot(A_dict['A'+str(i-1)], dK.T)
            dB_dict['dB'+str(i)] = np.sum(dK, axis = 1, keepdims=True)

        # Updates all the weights and biases
        # TODO:
        for i in range(1, node_num+1):
            # print(i)
            # print(f"weights{weights['W' + str(i)].shape} - alpha* dW_dict{dW_dict['dW' + str(i)].shape}")
            weights['W' + str(i)] = weights['W' + str(i)] - ALPHA*dW_dict['dW' + str(i)]
            biases['B' + str(i)] = biases['B' + str(i)] - ALPHA*dB_dict['dB' + str(i)]
    return weights, biases


def data_preprocessing(mode='train'):
    """
    :param mode: str, indicating if it's training mode or testing mode
    :return: Tuple(numpy_array, numpy_array), the first one is X, the other one is Y
    """
    data_lst = []
    label_lst = []
    first_data = True
    if mode == 'train':
        with open(TRAIN, 'r') as f:
            for line in f:
                data = line.split(',')
                # ['0PassengerId', '1Survived', '2Pclass', '3Last Name', '4First Name', '5Sex', '6Age', '7SibSp', '8Parch', '9Ticket', '10Fare', '11Cabin', '12Embarked']
                if first_data:
                    first_data = False
                    continue
                if not data[6]:
                    continue
                label = [int(data[1])]
                if data[5] == 'male':
                    sex = 1
                else:
                    sex = 0
                # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
                passenger_lst = [int(data[2]), sex, float(data[6]), int(data[7]), int(data[8]), float(data[10])]
                data_lst.append(passenger_lst)
                label_lst.append(label)
    else:
        pass
    return np.array(data_lst).T, np.array(label_lst).T


if __name__ == '__main__':
    main()
