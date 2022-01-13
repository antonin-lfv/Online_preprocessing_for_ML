import numpy as np
import pandas as pd
import binascii
from scipy.spatial import distance
from collections import Counter

def max_std(dataset):  # colonne de maximum de variance
    l = []
    for nom in dataset.columns:
        if type(dataset[nom][0]) != object and type(dataset[nom][0]) != str:
            l.append([dataset[nom].std(), nom])
    return max(l)


def col_numeric(df):  # retourne les colonnes numériques d'un dataframe
    return df.select_dtypes(include=np.number).columns.tolist()


def col_temporal(df):  # retourne les colonnes temporelles d'un dataframe
    return df.select_dtypes(include=np.datetime64).columns.tolist()


def clean_data(x):  # enlever les symboles d'une colonne
    if isinstance(x, str):
        return x.replace('$', '').replace(',', '').replace('€', '').replace('£', '')
    return x


def distance_e(x, y):  # distance entre 2 points du plan cartésien
    return distance.euclidean([x[0], x[1]], [y[0], y[1]])


def max_dist(donnee_apres_pca, df, voisins):  # pour knn, retourne la distance du voisin le plus loin
    distances = []
    for i in range(len(df)):
        distances.append(distance_e(donnee_apres_pca, [df['x'].iloc[i], df['y'].iloc[i]]))
    distances.sort()
    return distances[voisins-1]