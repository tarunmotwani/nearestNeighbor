import numpy as np
import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt  


def distance():


def nearestNeighbor():
    print("nearestNeighbor")
    


# def main():
    




if __name__ == '__main__':
    data = pd.read_csv('CS170_SMALLtestdata__6.txt', delim_whitespace=True, header = None, names = ['type', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    # arr = ['type', 'attr1', 'attr2', 'attr3']
    df = pd.DataFrame(data)
    print(df.tail())
    
    X = df.iloc[:, :-1].values                                  #pandas dataframe selections and indexing
    Y = df.iloc[]
    print(X[90])
