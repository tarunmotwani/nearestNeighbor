import numpy as np
import pandas as pd
import math
from math import sqrt 
import random
import copy
import time
import sys

def distance(p1,p2): #calculates the distance between two points
    dist = 0
    dist = float(pow(float(p1)-float(p2),2))
    return dist
    
def knn(data, i , attributes):  #data is entire dataset, i = each row, attribute = each column
    minDist, minDataArray, nearestIndex, count = 100000000, [], 0, 0#mindist is set high to find the low
    for j in range(len(data)): #for all rows in the dataset
        tempDist = 0 #reset distance
        for k in range(len(attributes)): #for all columns
            tempDist += distance(data[i][attributes[k]], data[j][attributes[k]]) #per attribute, calculating the distance between rows and summing them together
            tempDist = sqrt(tempDist)#taking the squared distance because we want positive numbers
        if i != j and tempDist < minDist: #never comparing the same two points
            minDist, nearestIndex = tempDist, j #updating minDist and its index
    return nearestIndex #return index of closest point
 
def intersection(currentFeatures, point): #calculates set intersection with respect to any point passed in
    for i in range(len(currentFeatures)): #for all current features
        if currentFeatures[i] == point: return True #check if any are the point
    return False#if none return false

def leave_one_out_cross_validation(data, attributes):
    count, result = 0,0
    for i in range(len(data)): #for all rows in data
        result = knn(data, i, attributes) #find each row's closest index
        if data[result][0] == data[i][0]: count += 1 #data is compared on the 1st column to check labels. +1 if match
    return (count/len(data))*100    #calculate the accuracy by correct/allrows*100 to find percentage

def forwardSelection(data): #finding accuracy starting from an empty set of current features
    df, currentSet, max, flag, bestSoFar = pd.DataFrame(data), [], -1, False,0 #pandas used for column function
    for i in range(len(df.columns)-1):  #for each level in the search tree
        array, featureToAdd, maxAccuracy, flag, localAccuracy = [], [], 0, False, 0 #resetting array for newCurrentset
        print("On level ", i+1, " Of the search tree")  #iterating through the levels of the search tree
        for k in range(1,len(df.columns)):  #for each attribute
            if not (intersection(currentSet, k)):   #if the currentSet intersects the current point, skip
                array = copy.deepcopy(currentSet) #copy currentSet to alter the contents
                array.append(k) #add current attribute
                accuracy = leave_one_out_cross_validation(data, array) #find accuracy of each attribute
                print("Using features(s) ", array, " accuracy is ", accuracy, "%")#display possible accuracy
                if accuracy > localAccuracy:    #compare accuracy to local to find local maxima
                    localAccuracy, localBestArray= accuracy,copy.deepcopy(array) #local finds best accuracy for each level of the tree
                if (accuracy > maxAccuracy):    #max accuracy comparison for all levels of the search tree
                    maxAccuracy, tempOfTemp, featureToAdd = accuracy, copy.deepcopy(array), k #find max and set true
        if bestSoFar < maxAccuracy: #comparator for local to best
            bestSoFar, bestTemp, flag = maxAccuracy, copy.deepcopy(tempOfTemp), True #finding max accuracy for whole function
        print("Feature to add: ", featureToAdd) #display the feature that has the most accuracy
        currentSet.append(featureToAdd) #append the feature to the currentset
        if i != len(df.columns)-2: #display only if we are still iterating otherwise pass
            if flag == False: print("Warning, accuracy has decreased! Continuing search in case of local maxima") #warning in case accuracy decreases over search tree
            print("Feature set ", localBestArray, "was best, with an accuracy of ", maxAccuracy, "%")   #best per level of the tree
    print("\nFinished Search!! The best feature subset is ", bestTemp,", which has an accuracy of ", bestSoFar, "%") #best overall

def backwardSelection(data, currentSet): #calculating accuracy from the full list of current features
    df, deletionArray, max, bestSoFar, tempOfTemp, bestTemp = pd.DataFrame(data), [], -1,0,0,0 #start max at -1 to ensure it increases
    for i in range(len(df.columns)-2):  #iterating through each column for the levels in the search tree
        print("On level ", i+1, " Of the search tree") #display each level of the search tree
        array, featureToRemove, minAccuracy, maxAccuracy, flag, localAccuracy = [], [], 1000000000, 0, False, 0 #init arrays temp min max and flags
        for k in range(1,len(df.columns)): #find the length of the arrays
            if not intersection(deletionArray, k):  #we do not want to use an element that has already been deleted
                array = copy.deepcopy(currentSet) #copying a currentSet to change
                array.remove(k) #removing an attribute to see if we gain any accuracy
                accuracy = leave_one_out_cross_validation(data, array) #find the accuracy with the feature removed
                print("Using features(s) ", array, " accuracy is ", accuracy, "%") #display accuracy
                if accuracy > localAccuracy: 
                    localBestArray = copy.deepcopy(array) #updating our local accuracy to display later
                if (accuracy > maxAccuracy): maxAccuracy, tempOfTemp, featureToRemove = accuracy, copy.deepcopy(array), k
        if len(currentSet) == 0: break #if currentset becomes empty we can finish the search
        deletionArray.append(featureToRemove)   #append deletion array with the next removal
        if bestSoFar < maxAccuracy: #chcking accuracy on all levels
            bestSoFar, bestTemp, flag = maxAccuracy,copy.deepcopy(tempOfTemp), True #updating the best accuracy for all levels
        print("Removed ", featureToRemove)  #display removal
        currentSet.remove(featureToRemove)#remove the intended feature  from the current set
        if i != len(df.columns)-2: #as long as we are not at the end, we have a warning message in case accuracy is decreasing
            if flag == False: print("Warning, accuracy has decreased! Continuing search in case of local maxima")
            print("Feature set ", localBestArray, "was best, with an accuracy of ", maxAccuracy, "%") #best per level of the tree
    print("\nFinished Search!! The best feature subset is ", bestTemp,", which has an accuracy of ", bestSoFar, "%") #best overall

def alphaBetaPruning(data):
    df, currentSet, max, flag, bestSoFar = pd.DataFrame(data), [], -1, False,0 #pandas used for column function
    for i in range(len(df.columns)-1):  #for each level in the search tree
        array, featureToAdd, maxAccuracy, flag, localAccuracy = [], [], 0, False, 0 #resetting array for newCurrentset
        print("On level ", i+1, " Of the search tree")  #iterating through the levels of the search tree
        for k in range(1,len(df.columns)):  #for each attribute
            if not (intersection(currentSet, k)):   #if the currentSet intersects the current point, skip
                array = copy.deepcopy(currentSet) #copy currentSet to alter the contents
                array.append(k) #add current attribute
                accuracy = leave_one_out_cross_validation(data, array) #find accuracy of each attribute
                print("Using features(s) ", array, " accuracy is ", accuracy, "%")#display possible accuracy
                if accuracy > localAccuracy:    #compare accuracy to local
                    localAccuracy, localBestArray= accuracy,copy.deepcopy(array) #local finds best accuracy for each level of the tree
                if (accuracy > maxAccuracy):    #max accuracy comparison for all levels of the search tree
                    maxAccuracy, tempOfTemp, featureToAdd = accuracy, copy.deepcopy(array), k #find max and set true
        if bestSoFar < maxAccuracy: 
            bestSoFar, bestTemp, flag = maxAccuracy, copy.deepcopy(tempOfTemp), True #finding max accuracy for whole function
        print("Feature to add: ", featureToAdd) #display the feature that has the most accuracy
        currentSet.append(featureToAdd) #append the feature to the currentset
        if i != len(df.columns)-2: #display only if we are still iterating otherwise pass
            if flag == False: 
                print("Accuracy has decreased... Ending Search") #warning in case accuracy decreases over search tree
                break #if accuracy decreases at all, we stop checking the worse cases after that
            print("Feature set ", localBestArray, "was best, with an accuracy of ", maxAccuracy, "%") #best per level
    print("\nFinished Search!! The best feature subset is ", bestTemp,", which has an accuracy of ", bestSoFar, "%") #best overall

if __name__ == '__main__':
    print("Welcome to Tarun Motwani's Feature Selection Algorithm ")
    # fileName = "CS170_SMALLtestdata__110.txt"
    fileName = input("Type in the name of the file to test: ")
    while len(fileName)<5:
        print("Error invalid input\n")
        fileName = input("Type in the name of the file to test: ")
    allFeatures =[]
    data = np.loadtxt(fileName)
    df = pd.DataFrame(data)
    for i in range(1,len(df.columns)):
        allFeatures.append(i)
    chooseAlg = input("Type the number of the algorithm you want to run. \n1) Forward Selection\n2) Backward Elimination\n3) Tarun’s Special Algorithm.\n")
    while not (chooseAlg == '1' or chooseAlg == '2' or chooseAlg == '3'):
        print("Error invalid input\n")
        chooseAlg = input("Type the number of the algorithm you want to run. \n1) Forward Selection\n2) Backward Elimination\n3) Tarun’s Special Algorithm.\n")
    print("\nThis dataset has", len(df.columns) - 1, "features (not including the class attribute, with", len(data), "instances")
    print("\nRunning nearest neighbor with all ",len(df.columns)-1," features, using “leaving-one-out” evaluation, I get an accuracy of ", leave_one_out_cross_validation(data, allFeatures) , "%")
    print("\nBeginning Search.")
    # print(data)
    if chooseAlg == '1': forwardSelection(data)
    elif chooseAlg == '2': backwardSelection(data, allFeatures)
    else: alphaBetaPruning(data)        
    sys.exit()