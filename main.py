import numpy as np
import pandas as pd
import math
from math import sqrt 

#https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/?fbclid=IwAR373As3huplgQOcQDoMJ6AeqGYl1ujuhJD5QA7RtrDxM1NCony4kWlB5F0
p1 = 0

def distance(p1,p2):
    dist = 0
    for i in range(len(p1)): dist += float(pow(float(p1[i])-float(p2[i]),2))
    return sqrt(dist)

def nearestNeighbor(X):
    minDist = distance(X[0], X[1])
    for i in range(len(X)):
        for j in range(len(X)):
            tempDist = distance(X[i], X[j])
            if i != j and minDist > tempDist: minDist = tempDist    
    return minDist

# def feature_search(X):
#specify for one attribute using distance(p1[0], p2[0])
# (p1[:1], p2[])
# def feature_search(X):
#     current = []
#     for i in len(X):
#         print('on the ')


if __name__ == '__main__':
    print("Welcome to Tarun Motwani's Feature Selection Algorithm ")
    # fileName = input("Type in the name of the file to test: ")
    print("Type the number of the algorithm you want to run. \n1) Forward Selection\n2) Backward Elimination\n3) Tarun’s Special Algorithm.\n")
    data = pd.read_csv('CS170_SMALLtestdata__6.txt', delim_whitespace=True, header = None, names = ['type', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    df = pd.DataFrame(data)
    # print(df.head(20))
    
    X = df.iloc[:, 1:].values                                   #pandas dataframe selections and indexing
    Y = df.iloc[:, 0].values                                    #class type

    #testing
    print(distance(X[2], X[4]))
    # print(nearestNeighbor(X))
