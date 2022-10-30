#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    # x = x.strip()
    # x = x.lower()

    ## Ver 1.
    # FeatureVector = {}
    # for i in x:
    #     if i in FeatureVector:
    #         FeatureVector[i] += 1
    #     else:
    #         FeatureVector[i] = 1

    ## Ver 2. Using Defaultdict
    x = x.split(" ")
    FeatureVector = defaultdict(int)
    for i in x:
        FeatureVector[i] += 1
    return FeatureVector
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE


############################################################
# Milestone 4: Sentiment Classification

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float):
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    """
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def predictor(y):
        print(y)
        training_error = evaluatePredictor(trainExamples, y)
        validation_error = evaluatePredictor(validationExamples, y)
        print(f'Training Error ({epoch} Epoch): {training_error}')
        print(f'Validation Error ({epoch} Epoch): {validation_error}')

    ## Test parameter
    # numEpochs = 1
    ##
    for epoch in range(numEpochs):
        # cost = 0
        for x, y in trainExamples:
            # y = 0, if y <0 else 1
            phi_x = featureExtractor(x)
            if dotProduct(weights, phi_x) *y >= 1:
                pass
            else:
                # w = w -alpha * (-phi_x)* y
                increment(weights, alpha*y, phi_x)


        # for i in trainExamples:
        #     feature = featureExtractor(i[0])
        #     y = 1 if i[1] == 1 else 0
        #     h = 1/(1 + math.exp(-(dotProduct(weights,feature))))
        #     # cost = (-(y*math.log(h)+(1-y)*math.log(1-h)))
        #     # print(f'Cost: {cost} Epoch: {epoch}')
        #     weights = increment(weights,-alpha*(h-y),feature)
        # # Predictor First x: 1 or -1 ; Second x: trainExamples[0], and dot weights in each epoch
        # predictor(lambda x : (1 if dotProduct(featureExtractor(x), weights) > 0 else -1))


    # END_YOUR_CODE
    return weights


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrance.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)

        # 計算 weight 內word數量 phi = {word : 隨機word數量}
        phi = defaultdict(int)
        for _ in range(random.randint(1, 100)):
            word = random.choice(list(weights))
            phi[word] += 1
        # phi 與 weights內積
        y = 1 if dotProduct(phi, weights) >= 0 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        dict = defaultdict(int)
        x = x.replace(" ", "")
        for i in range(len(x)-n+1):
            word = ""
            for j in range(n):
                word += x[i+j]
            dict[word] += 1
        return dict
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))


## Test
# trainExamples = (("hello world", 1), ("goodnight moon", -1))
# testExamples = (("hello", 1), ("moon", -1))
# featureExtractor = extractWordFeatures
# weights = learnPredictor(trainExamples, testExamples, featureExtractor, numEpochs=20, alpha=0.01)
# print(weights)

