# File: TaylorBrady-L05-CategoryData.py
# Author: Taylor Brady
# Date: 8/10/2020
# Other comments:
# Summary
#   Categorical variables did not need to be decoded
#   Missing categorical variables were replaced with the most common value
#   Capital-gain and Capital-loss (numeric) were consolidated by subtracting 
#       the loss from the gain
#   Relationship and marital-status were consolidated into a boolean for 
#       wether or not the individual is married
#   Native country was consolidated by combining categories into regions
#   Non-boolean categories were one-hot encoded using the getdummies function
#   Every attribute is plotted. After consolidation, but before one-hot 
#       encoding. 
#   The scatter matrix plots numerical data, but is color coded for the 
#       predicted value column


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


# Description: Use Z-normalization to normalize column
def normalizeData(data):
    m = data.mean()
    data = data - m
    s = data.std()
    data = data / s
    return data


# Description: Displays plot of data, uses histogram for numerical,
#   and bar for categorical
def view(data, numBins=10, t = ''):
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
def fill_missing(data):
    if data.dtype.name == 'category':
        v = data.value_counts().nlargest(n=1).index[0]
        data = data.str.replace(r'.*\?.*', v, regex=True)
    else:
        data = pd.to_numeric(data, errors='coerce')
        m = data.mean()
        data = data.replace(float("NaN"), m)
    return data


if __name__ == "__main__":
    # Import data
    adult = importData()

    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country',
                'predicted-earning']
    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                'capital-loss', 'hours-per-week']
    # enforce categorical datatype
    adult_nums = adult.loc[:, num_cols]
    adult_cats = adult.loc[:, cat_cols].astype('category')

    # consolidate capital-loss & capital gain
    gain = adult.loc[:, 'capital-gain'] - adult.loc[:, 'capital-loss']
    adult_nums['capital-gain'] = gain
    adult_nums.pop('capital-loss')



    adult_cats.pop('relationship')

    for key, value in adult_nums.iteritems():
        # clean up missing values
        value = fill_missing(value)
        value = normalizeData(value)
        # value = binData(value)
        adult_nums.loc[:, key] = value

    for key, value in adult_cats.iteritems():
        value = fill_missing(value)
        adult_cats.loc[:, key] = value

    # consolidate countries by region
    temp = adult_cats.loc[:, 'native-country']

    caribbean = [' Cuba', ' Jamaica', ' Haiti',
                 ' Dominican-Republic', ' Trinadad&Tobago']
    Us_terrs = [' Puerto-Rico', ' Outlying-US(Guam-USVI-etc)']
    s_c_am = [' Honduras', ' El-Salvador', ' Guatemala', ' Columbia',
              ' Ecuador', ' Peru', ' Nicaragua']
    Europe = [' Italy', ' Portugal', ' England', ' Germany', ' Poland',
              ' France', ' Yugoslavia', ' Greece', ' Scotland', ' Ireland',
              ' Holand-Netherlands', ' Hungary']
    asia = [' India', ' China', ' Japan', ' Hong', ' Cambodia', ' Thailand',
            ' Laos', ' Vietnam', ' Taiwan', ' Philippines', ' Iran']
    n_am = [' Mexico', ' Canada']

    temp = temp.replace(caribbean, 'Caribbean')
    temp = temp.replace(Us_terrs, 'US_Territories')
    temp = temp.replace(s_c_am, 'South-Central_America')
    temp = temp.replace(Europe, 'Europe')
    temp = temp.replace(asia, 'Asia')
    temp = temp.replace(n_am, 'North_America')

    adult_cats.loc[:, 'native-country'] = temp.copy()

    # consolidate marital status and relationship into married/single
    temp = ((adult.loc[:, 'marital-status'] == ' Married-civ-spouse') |
               (adult.loc[:, 'marital-status'] == ' Married-AF-spouse'))
    adult_cats.loc[:, 'marital-status'] = temp.copy()
    adult_cats = adult_cats.rename(columns={'marital-status': 'is_married',
                                            'native-country': 'region'})
    adult_cats['is_married'][temp == True] = 1
    adult_cats['is_married'][temp != True] = 0
    # Turn sex into 1/0
    temp = adult_cats.loc[:, 'sex'] == ' Male'
    adult_cats.pop('sex')
    adult_cats['is_male'] = temp.copy()
    adult_cats['is_male'][temp == True] = 1
    adult_cats['is_male'][temp != True] = 0
    temp = adult_cats.loc[:, 'predicted-earning'] == ' >50K'
    adult_cats.pop('predicted-earning')
    adult_cats['predicted-high'] = temp.copy()
    adult_cats['predicted-high'][temp == True] = 1
    adult_cats['predicted-high'][temp != True] = 0    
    
    adult_cats = adult_cats.astype('category')
    # Display plots
    for key, value in adult_nums.iteritems():
        view(adult_nums.loc[:, key])
    for key, value in adult_cats.iteritems():
        view(adult_cats.loc[:, key], t='category')

    scatter_matrix(adult_nums, alpha=0.1, s=3, c=adult_cats.loc[:, 'predicted-high'])
    # one-hot encode categorical data
    for key in ['workclass', 'education', 'occupation', 'race', 'region']:
        adult_cats = adult_cats.join(adult_cats.loc[:, key].str.get_dummies())
        adult_cats.pop(key)
