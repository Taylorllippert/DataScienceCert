#File: TaylorBrady-<lesson number>-<assignment name>.py
#Description: 
#Author: Taylor Brady
#Date: 7/17/2020
#Other comments: 

    
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
import numpy as np

#Description: 
def importData():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    data = pd.read_csv(url, header=None)
    data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'predicted-earning']
    return data


#Description:
def cats_to_dums(data):
    x = data.str.get_dummies()
    return x


#Description:
def normalizeData(data):
    m = data.mean()
    s = data.std()
    data = data - m
    data = data / s
    return data


def view(data):
    if data.dtype.name == 'cateogory':
        data.value_counts().plot(kind='bar')
    else:
        plt.hist(data)
    plt.title(data.name)
    plt.show()



    
def binData(data, numBins = 5):
    bData = np.empty(len(data), int)
    bwidth = (max(data)-min(data))/numBins
    b = np.linspace(np.min(data), np.max(data), numBins + 1) 
    for i in range(1, numBins):
        bData[(data >= b[i-1]) & (data < b[i])] = i
    bData[data == b[-1]] = numBins -1
        
    return bData

#Description:
def fill_missing(data):
    if data.dtype.name == 'category':
        v = data.value_counts().nlargest(n=1).index[0]
        print(data[data==' ?'].count())
        data = data.str.replace('.*\?.*', v, regex=True)
        print(data[data==' ?'].count())
        
    else:
        data = pd.to_numeric(data, errors = 'coerce')
        m = data.mean()
        data = data.replace(float("NaN"), m)
    return data

#Description: 
if __name__ == "__main__":
    adult = importData()

    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                'relationship', 'race', 'sex', 'native-country', 'predicted-earning']
    adult.loc[:, cat_cols] = adult.loc[:, cat_cols].astype('category')
    for key, value in adult.iteritems():
        value = fill_missing(value)
        if adult.loc[:, key].dtype.name == 'category':
            view(value)
        else:
            value = normalizeData(value)
            view(value)
