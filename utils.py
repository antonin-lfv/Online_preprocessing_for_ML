import binascii
from scipy.spatial import distance
from collections import Counter
import numpy as np
import pandas as pd
import time
import itertools
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import binascii
import os
import webbrowser
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LinearRegression, PoissonRegressor, ElasticNet, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, scale, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn import metrics
from sklearn.metrics import *
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
import umap.umap_ as umap
from scipy.spatial import distance
import more_itertools
from streamlit_lottie import st_lottie
import requests
from sklearn.pipeline import make_pipeline

CSS = """
<style>
.first_titre {
    font-size:75px !important;
    font-weight: bold;
    box-sizing: border-box;
    text-align: center;
    width: 100%;
}
.intro{
    text-align: justify;
    font-size:20px !important;
}
.grand_titre {
    font-size:40px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #4976E4;
    text-decoration-thickness: 5px;
    width: 100%;
}
.section{
    font-size:20px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #111111;
    text-decoration-thickness: 3px;
}
.petite_section{
    font-size:16px !important;
    font-weight: bold;
}
.nom_colonne_page3{
    font-size:17px !important;
    text-decoration: underline;
    text-decoration-color: #000;
    text-decoration-thickness: 1px;
}
</style>
"""

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

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()