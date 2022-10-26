import math

def sigmoid(k):
    # print(1/1+(math.exp(-k)))
    return 1/(1+(math.exp(-k)))

def loss(y,h):
    return -(y*math.log(h)+(1-y)*math.log(1-h))

print(loss(0, sigmoid(-2)))