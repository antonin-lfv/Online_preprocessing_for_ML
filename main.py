""" preprocessing on a dataframe """
# streamlit run main.py #

import numpy as np
from collections import Counter
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

####### Streamlit Config ######
st.set_page_config(layout="wide", )
st.title('Preprocessing automatique')

filename = st.sidebar.text_input('Enter a file path:')
if filename=='':
    pass
else :
    try:
        with open(filename) as input:
            data = pd.read_csv(filename)
            st.sidebar.success('Fichier chargé')


            st.text(data.columns)
    except FileNotFoundError:
        st.sidebar.error('Fichier non trouvé')


## have information on a series

def column_info(data): # data -> 1 serie
    (a,) = data.shape
    if (type(data)==pd.core.series.Series):
        n_data = (data.to_numpy())
    elif (type(data)==np.ndarray):
        n_data = data
    else :
        return 'error type_data'
    print(' ● données:')
    print(data.head(5))
    print('-------------------------')
    time.sleep(1)
    print(' ● data type :', type(data))
    print('-------------------------')
    time.sleep(2)
    if n_data.dtype == float :
        moyenne = data.mean()
        variance = data.std()
        max = data.max()
        min = data.min()
        print(' ● average :', moyenne)
        print('-------------------------')
        time.sleep(1)
        print(' ● variance :', variance)
        print('-------------------------')
        time.sleep(1)
        print(' ● maximum :', max)
        print('-------------------------')
        time.sleep(1)
        print(' ● minimum :', min)
        print('-------------------------')
        time.sleep(1)
    print(' ● most present value:',(Counter(n_data).most_common()[0])[0],':',(Counter(n_data).most_common()[0])[1],'fois',',',(Counter(n_data).most_common()[1])[0],':',(Counter(n_data).most_common()[1])[1],'fois')
    print('-------------------------')
    time.sleep(1)
    print(' ● missing values:', sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))
    print('-------------------------')
    time.sleep(1)
    print(' ● length:',n_data.shape[0])
    print('-------------------------')
    time.sleep(1)
    print(' ● number of different values not NaN:', abs(len(Counter(n_data))-sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
    print('-------------------------')
    time.sleep(1)

## have information on the entire dataframe

def max_std(dataset):
    l = []
    for nom in dataset.columns :
        if type(dataset[nom][0]) != object and type(dataset[nom][0]) != str :
            l.append([dataset[nom].std(),nom])
    return(max(l))

#dataset with type Dataframe :
# " Numpy -> pandas.Dataframe : pd.DataFrame()"

def dataset_info(dataset): #type(nom_col_index)=str
    print('')
    print(' ● size:', dataset.shape)
    print('-------------------------')
    time.sleep(0.5)
    print(' ● data type:', dataset.dtypes.value_counts())
    print('-------------------------')
    time.sleep(0.5)
    print(' ● missing values:', sum(pd.DataFrame(dataset).isnull().sum(axis=1).tolist()))
    print('-------------------------')
    time.sleep(0.5)
    print(' ● number of values:', dataset.shape[0]*dataset.shape[1])
    print('-------------------------')
    time.sleep(0.5)
    print(' ● maximum variance for the column :', max_std(dataset)[1])
    print('   with a variance of :',max_std(dataset)[0] )
    print('-------------------------')
    if sum(pd.DataFrame(dataset).isnull().sum(axis=1).tolist()) != 0 :
        sns.heatmap(dataset.isna(),cbar=False)
        plt.title('brigth color = no data') #if missing data we plot the dataset
    dataset[[str(max_std(dataset)[1])]].plot()
    plt.title('most progressive feature')
    plt.show()

