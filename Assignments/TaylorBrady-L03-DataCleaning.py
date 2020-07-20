#File: TaylorBrady-L03-DataCleaning.py
#Description: 
#Author: Taylor Brady
#Date: 7/17/2020
#Other comments: 
    
# =============================================================================

# =============================================================================

import numpy as np
import random as rand

#Description: removes outliers in arr1
#Pre: arr is type ndarray and contains only numeric items
#Post: items that are not within 2 standard deviations of the mean are removed
#       and the array is returned
def remove_outlier(arr):
    try:
        arr = arr[~find_outlier(arr)]
    except:
        print("An exception has occured")
    return arr



#Description: finds outliers in an array
#pre array is type ndarray and contains only numeric values
#post: boolean array is returned that values true when values are outliers
def find_outlier(arr):
    try:
# flags outliers as numbers outside of 2 standard deviations from the mean
        m = arr.mean()
        s = arr.std()
        limitLo = m - (2 * s)
        limitHi = m + (2 * s)
        flagGood = (arr >= limitLo) & (arr <= limitHi)
        return ~flagGood
    except:
        print("An exception has occured")
    return 1
    


#Description: replaces outliers in arr1 with the arithmetic mean of the non-outliers
#Pre: array is type ndarray and contains only numeric values
#Post: items that are not within 2 standard deviations of the mean are replaced 
#       with the mean (without the outliers) and the array is returned
def replace_outlier(arr):
    try:
        flagBad=find_outlier(arr)
# computes the mean of the non-outlier values
        m = arr[~flagBad].mean()
# replaces outliers with mean        
        arr[flagBad] = m
    except:
        print("An exception has occured")
    return arr


#Description: fills in the missing values in arr2 with the median of arr2
#Pre: array is type ndarray, can contain non-numeric values
#Post: non numeric values are replaced with the median value and the array is retured
def fill_median(arr):
    try:
# creates boolean flag that is true for digits and false for non-numeric values
        flagGood=np.array([element.isdigit() for element in arr])
# computes the median of the numeric values        
        m= np.median(arr[flagGood].astype(int))
# replaces the non-numeric values and converts the array to int        
        arr[ ~flagGood ]=m
        arr = arr.astype(int)
    except:
        print("An exception has occured")
    return arr
    
    
#Description: Creates test arrays with outliers and non-numeric values
#               calls 'cleaning' functions
if __name__ == "__main__":
    
    rand.seed()
    x = np.arange(10,101)
    
 #Create a numeric numpy array, named arr1, with at least 30 items that contains outliers
    arr1 = np.array(rand.choices(x,k=40))
    arr1[rand.randrange(0,39)]=5000
    arr1[rand.randrange(0,39)]=5000
    
    
#arr2 contains improper non-numeric missing values, like "?"
    arr2 = np.array(rand.choices(x,k=40), dtype=str)
    arr2[rand.randrange(0,39)]=  "?"
    arr2[rand.randrange(0,39)]= ' '
    

    print("Original Arr1:")
    print(arr1)
    
    
    print("Arr1 with outliers removed:")
    print(remove_outlier(arr1))
    
    print("Arr1 with outliers replaced:")
    print(replace_outlier(arr1))
    
    
    print("Original Arr2:")
    print(arr2)
    
    print("Arr2 with outliers non-numeric values replaced:")
    print(fill_median(arr2))
    
    
