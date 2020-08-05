#File: TaylorBrady-L04-NumericData.py
#Description: Begining EDA
#Author: Taylor Brady
#Date: 8/3/2020
#Other comments:

# Code snippet:
#           with warnings.catch_warnings():
#              warnings.simplefilter(action='ignore', category=FutureWarning)
# source: https://stackoverflow.com/questions/40659212/futurewarning-
#   elementwise-comparison-failed-returning-scalar-but-in-the-futur
# purpose: warning that comparison may change from 0/1 to False/True, but this
#   change does not change the functionality of the lines that are throwing the warning

# =============================================================================
#               summary comment block
# Which attributes had outliers and how were the outliers defined?
#   Outliers are defined as values more than 2 standard deviations away from the
#       mean.
#   Attributes containing outliers, and count of outliers:
#       age(14), trestbps(14), chol(14), thalach(6), oldpeak(14)
#
# Which attributes required imputation of missing values and why?
#   Missing values were replaced by the median value of the attributes in order to
#       histogram the data and preform analysis on the data
#   Attributes with were missing values were replaced, and count:
#       trestbps(1), chol(23), thalach(1)
#
# Which attributes were histogrammed and why?
#   The non-categorical, numerical attributes were histogrammed. These include:
#       age, trestbps, chol, thalach, and oldpeak
#
# Which attributes were removed and why?
#   Attributes that had over 50% of their values missing were removed.
#   The attributes and the percent of missing values are:
#       slope(65%), ca(99%), thal(90%)
#
# How did you determine which rows should be removed?
#   Tuples that had missing categorical data (after the attributes with missing
#       data were removed), were removed.
#       10 tuples were removed.
# =============================================================================

import warnings
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

plt.set_cmap('coolwarm')



#Description: replaces missing values
#Pre: data is a series
#Post: data contains only numeric values and '?', '?' values are replaced with the median value
def replace_missing(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            empty = data.loc[:] == '?'
        #How many missing values
        #print("Replaced " + str(empty.sum()) + " missing values in " + data.name)
        data[empty] = pd.to_numeric(data[~empty]).median()
        data = pd.to_numeric(data)
    except:
        print("Error in replace missing values")
    return data


#Description:Replaces outliers with min/max values
#Pre: data has 'convertable' numeric values and '?' only
#Post: data is returned with replaced outliers
def replace_outliers(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            d = pd.to_numeric(data[(data != '?')])
        m = d.mean()
        s = d.std()
        h = m + (2 * s)
        l = m - (2 * s)
        #How many outliers
        #good=(d>l) & (d < h)
        #print("Replaced " +str(d.count() - good.sum())+ "outliers in" + data.name)
        d[(d < l)] = l
        d[(d > h)] = h
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            data[(data != '?')] = d.copy()
    except:
        print("Error in replace outliers")
    return data

#Description: Displays histogram of data and prints the standard deviation
#Pre: data contains 'convertable' numeric values, and has name property
#Post: data is not changed
def display_data(data):
    try:
        plt.hist(pd.to_numeric(data))
        plt.title(data.name)
        plt.show()
        print("Std of " +data.name+" : "+str(pd.to_numeric(data).std()))
    except:
        print("Error in display data")



#Description: imports data from url and assigns column names
#Pre: Data set is located at url, and url is reachable
#Post: Dataframe with imported data, and appropriate column names is returned
def importData():
    # The url for the data
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
    data = pd.read_csv(url, header=None)
    # Replace the default column names
    data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                    "restecg", "thalach", "exang", "oldpeak", "slope",
                    "ca", "thal", "num"]
    return data

#Description: Imports dataset, cleans data, and displays data
if __name__ == "__main__":
    heart = importData()
    heart.loc[:, 'sex'] = heart.loc[:, 'sex'].astype('category')
    heart.loc[:, 'cp'] = heart.loc[:, 'cp'].astype('category')
    heart.loc[:, 'fbs'] = heart.loc[:, 'fbs'].astype('category')
    heart.loc[:, 'restecg'] = heart.loc[:, 'restecg'].astype('category')
    heart.loc[:, 'exang'] = heart.loc[:, 'exang'].astype('category')
    heart.loc[:, 'num'] = heart.loc[:, 'num'].astype('category')
    origCount = heart.shape[0]


    for key, value in heart.iteritems():
        if heart.loc[:, key].dtype.name == 'object':
            # Removes attributes that are mostly missing
            if value[(value == '?')].count()/value.count() > .5:
                #Shows which attributes are removed and the percent of missing values
                #print(key + " removed, "+ str(value[(value=='?')].count()/value.count())+
                #"% missing")
                heart.pop(key)
                continue
    for key, value in heart.iteritems():
        # Remove tuples with missing categorical data
        if heart.loc[:, key].dtype.name == 'category':
            tup=heart.loc[:, key][(heart.loc[:, key] == '?')]
            if tup.count() != 0:
                #Shows how many missing values were contained in each tuple about to be dropped
                #print(str(tup.count()) + " tuples containing " +
                #str(heart.loc[tup.index,:][(heart=='?')].count(axis=1).sum()
                #                             ) + " missing value are dropped")
                heart = heart.drop(tup.index)
        else:
            heart.loc[:, key] = replace_outliers(heart.loc[:, key].copy())
            heart.loc[:, key] = replace_missing(heart.loc[:, key].copy())
            display_data(heart.loc[:, key])
    sex = heart.loc[:, 'sex']
    #Shows how many tuples are removed
    #print(str(origCount-heart.shape[0])+ " tuples removed")
    scatter_matrix(heart, c=sex, s=5)