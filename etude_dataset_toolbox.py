""" Etude d'un dataset """

import numpy as np
from collections import Counter
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

## avoir des renseignement sur une colonne

def column_info(data): # data -> 1 colonne
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
    print(' ● type de données :', type(data))
    print('-------------------------')
    time.sleep(2)
    if n_data.dtype == float :
        moyenne = data.mean()
        variance = data.std()
        max = data.max()
        min = data.min()
        print(' ● moyenne :', moyenne)
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
    print(' ● valeur les plus présentes:',(Counter(n_data).most_common()[0])[0],':',(Counter(n_data).most_common()[0])[1],'fois',',',(Counter(n_data).most_common()[1])[0],':',(Counter(n_data).most_common()[1])[1],'fois')
    print('-------------------------')
    time.sleep(1)
    print(' ● valeurs manquantes:', sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))
    print('-------------------------')
    time.sleep(1)
    print(' ● longueur:',n_data.shape[0])
    print('-------------------------')
    time.sleep(1)
    print(' ● nombre de valeurs distinctes non NaN:', len(Counter(n_data))-sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))
    print('-------------------------')
    time.sleep(1)


## avoir des renseignements sur le dataset entier

#dataset de type Dataframe :
" Numpy -> pandas.Dataframe : pd.DataFrame()"

def dataset_info(dataset):
    print(' ● taille:', dataset.shape)
    print('-------------------------')
    time.sleep(1)
    print(' ● types de données:', dataset.dtypes.value_counts())
    print('-------------------------')
    time.sleep(2)
    print(' ● valeurs manquantes:', sum(pd.DataFrame(dataset).isnull().sum(axis=1).tolist()))
    print('-------------------------')
    time.sleep(2)
    print(' ● nombre de valeurs:', dataset.shape[0]*dataset.shape[1])
    sns.heatmap(dataset.isna(),cbar=False)
    plt.title('dataset vizualisation')
    plt.show()
