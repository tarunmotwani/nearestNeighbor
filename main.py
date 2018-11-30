import numpy as np
import pandas as pd
import math
from math import sqrt 
import random
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut  
from sklearn import metrics


#https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/?fbclid=IwAR373As3huplgQOcQDoMJ6AeqGYl1ujuhJD5QA7RtrDxM1NCony4kWlB5F0

def distance(p1,p2):
    dist = 0
    for i in range(len(p1)): dist += float(pow(float(p1[i])-float(p2[i]),2))
    return sqrt(dist)

def nearestNeighbor(X):
    minDist = 10000
    minDataArray = []
    for j in range(len(X)):
        for k in range(len(X)):
            tempDist = distance(X[j], X[k])
            if j != k and minDist > tempDist: 
                minDist = tempDist
        minDataArray.append(minDist)
    return minDataArray    

# def feature_search(X):
#specify for one attribute using distance(p1[0], p2[0])
# (p1[:1], p2[])
def intersection(currentFeatures, point):
    # print(lst1, p2)
    for i in range(len(currentFeatures)):
        if currentFeatures[i] == point:
            return True
    return False

def feature_search(data):
    currentFeatures = []
    accuracy = 0
    # print("word", intersection(currentFeatures, 2))
    for i in range(len(data.columns)-1):
        # print("-----------------------------------------------------------------",currentFeatures)
        featureToAdd = 0
        bestAccuracy = 0
        print("on the %dth level of the search tree"%(i+1))
        for j in range(len(data.columns)-1):
            if not intersection(currentFeatures, j):
                print("Considering adding the %dth feature"%(j+1))
                accuracy = leave_one_out_cross_validation(data, currentFeatures, j)
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    featureToAdd = j
        currentFeatures.append(featureToAdd)
        # print(currentFeatures)
        print("On level %d, I added feature %s to current set"%(i+1, featureToAdd+1))
    print(currentFeatures)
    return
def leave_one_out_cross_validation(data, currentSet, featureToAdd):
    X = data.columns
    
    return random.randint(1,101)




if __name__ == '__main__':
    print("Welcome to Tarun Motwani's Feature Selection Algorithm ")
    # fileName = input("Type in the name of the file to test: ")
    print("Type the number of the algorithm you want to run. \n1) Forward Selection\n2) Backward Elimination\n3) Tarun’s Special Algorithm.\n")
    data = pd.read_csv('CS170_SMALLtestdata__6.txt', delim_whitespace=True, header = None, names = ['type', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    df = pd.DataFrame(data)
    # print(df.head(20))
    # print(df['a'])
    X = df.iloc[:, 1:].values                                   #pandas dataframe selections and indexing
    Y = df.iloc[:, 0].values                                    #class type
    # print(df.iloc[:, 1].values)

    print("This dataset has", len(df.columns) - 1, "features (not including the class attribute, with", len(X), "instances")
    print("Beginning Search.")
    
    #testing
    # print(distance(X[2], X[4]))
    # print("Running nearest neighbor with %d, we get an accuracy of"%(len(df.columns)-1), "", nearestNeighbor(X))
    feature_search(df)
    # leave_one_out_cross_validation(X)
