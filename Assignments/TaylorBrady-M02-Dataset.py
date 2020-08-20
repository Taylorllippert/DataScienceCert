# File: TaylorBrady-M02-Dataset.py
# Author: Taylor Brady
# Date: 8/18/2020
# Other comments: Improved data clean up from L05 assignment


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix

plt.set_cmap('coolwarm')


# Description: Imports dataset from source on the internet, and assigns column
#   headers
def importData():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    data = pd.read_csv(url, header=None, na_values='?')
    data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race',
                    'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'predicted-earning']
    return data


# Description: bound data to remove outliers
def bound(data):
    med = data.median()
    sdev = data.std()
    high = med + (2 * sdev)
    low = med - (2 * sdev)
    data[data > high] = high
    data[data < low] = low
    return data


# Description: Use Z-normalization to normalize column
def normalize(data):
    m = data.mean()
    data = data - m
    s = data.std()
    data = data / s
    return data


# Description: Displays plot of data, uses histogram for numerical,
#   and bar for categorical
def view(data, numBins=10, t=''):
    if t == 'category':
        data.value_counts().plot(kind='bar')
    else:
        plt.hist(data, bins=numBins)
        plt.xlabel('Distribution after Normalization and Binning')
    plt.ylabel('Frequency')
    plt.title(data.name)

    plt.show()


# Description: Returns binned data, where each bin is equal width
def binData(data, numBins=10):
    bData = data.copy()
    b = np.linspace(np.min(data), np.max(data), numBins + 1)
    for i in range(1, numBins+1):
        bData[(data >= b[i-1]) & (data < b[i])] = i
    bData[data == b[-1]] = numBins - 1
    return bData


# Description: Replaces missing values with the mean for numerical data
#   and the most popular value for categorical data
def fill_missing(data, t=''):
    if t == 'category':
        v = data.value_counts().nlargest(n=1).index[0]
        data = data.str.replace(r'.*\?.*', v, regex=True)
    else:
        data = pd.to_numeric(data, errors='coerce')
        m = data.mean()
        data = data.replace(np.nan, m)
    return data


def handle_numeric(data):
    for key, value in data.iteritems():
        data.loc[:, key] = fill_missing(data.loc[:, key].copy())
        data.loc[:, key] = bound(data.loc[:, key].copy())
        data.loc[:, key] = normalize(data.loc[:, key].copy())
        NB = 50
        data.loc[:, key] = binData(data.loc[:, key], numBins=NB)
    return data


def consolidate_cols(data, trans, default='?'):
    for key in trans:
        data = data.replace(trans[key], key)
    data[~data.isin(trans.keys())] = default
    return data


if __name__ == "__main__":
    # Import data
    adult = importData()

    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                'hours-per-week']

    # consolidate capital-loss & capital gain
    adult['capital-gain'] -= adult.loc[:, 'capital-loss']
    adult.pop('capital-loss')
    adult.loc[:, num_cols] = handle_numeric(adult.loc[:, num_cols])

    # consolidate countries by region
    Country_to_Region = {
        'Carribean': [' Cuba', ' Jamaica', ' Haiti', ' Dominican-Republic',
                      ' Trinadad&Tobago'],
        'US_Territories': [' Puerto-Rico', ' Outlying-US(Guam-USVI-etc)'],
        'South-Central_America': [' Honduras', ' El-Salvador', ' Guatemala',
                                  ' Columbia', ' Ecuador', ' Peru',
                                  ' Nicaragua'],
        'Europe': [' Italy', ' Portugal', ' England', ' Germany', ' Poland',
                   ' France', ' Yugoslavia', ' Greece', ' Scotland',
                   ' Ireland', ' Holand-Netherlands', ' Hungary'],
        'Asia': [' India', ' China', ' Japan', ' Hong', ' Cambodia',
                 ' Thailand', ' Laos', ' Vietnam', ' Taiwan', ' Philippines',
                 ' Iran'],
        'North_America': [' Mexico', ' Canada'],
        'United_States': ' United-States',
        'South': ' South',
        '?': ' ?'}
    adult.loc[:, 'native-country'] = consolidate_cols(
        adult.loc[:, 'native-country'], Country_to_Region)
    # consolidate/simplify marital statis /relationship
    Simplify_Marital_Status = {
        1: [' Married-civ-spouse', ' Married-AF-spouse'], '?': ' ?'}
    adult.loc[:, 'marital-status'] = consolidate_cols(
        adult.loc[:, 'marital-status'], Simplify_Marital_Status, default=0)
    adult.pop('relationship')
    # simplify sex
    Simplify_Gender = {
        1: ' Male', 0: ' Female'}
    adult.loc[:, 'sex'] = consolidate_cols(
        adult.loc[:, 'sex'], Simplify_Gender)
    # simplify predicted value
    Simplify_Prediction = {
        1: ' >50K', 0: ' <=50K'}
    adult.loc[:, 'predicted-earning'] = consolidate_cols(
        adult.loc[:, 'predicted-earning'], Simplify_Prediction)
    # rename columns that have been simplified and consolidated
    adult = adult.rename(columns={'marital-status': 'is_married',
                                  'native-country': 'region', 'sex': 'is_male',
                                  'predicted-earning': 'predicted_high'})
    # separate boolean columns from categorical
    cat_cols = ['workclass', 'occupation', 'race', 'region']
    bool_cols = ['is_married', 'is_male', 'predicted_high']

    # fill missing values for non-numeric attributes
    for key in cat_cols:
        adult.loc[:, key] = fill_missing(
            adult.loc[:, key].copy().astype('str'), t='category')

    for key in bool_cols:
        adult.loc[:, key] = fill_missing(
            adult.loc[:, key].copy().astype('str'), t='category')

    adult.loc[:, cat_cols] = adult.loc[:, cat_cols].astype('category')
    adult.loc[:, bool_cols] = adult.loc[:, bool_cols].astype('int')
    # Display plots
    for key in num_cols:
        view(adult.loc[:, key])
    for key in cat_cols:
        view(adult.loc[:, key], t='category')

    scatter_matrix(adult.loc[:, num_cols], alpha=0.05, s=5,
                   c=adult.loc[:, 'predicted_high'])
    # one-hot encode categorical data
    for key in cat_cols:
        adult = adult.join(adult.loc[:, key].str.get_dummies())
        adult.pop(key)

    adult.to_csv('TaylorBrady-M02-Dataset.csv', index=False)
