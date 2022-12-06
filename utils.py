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
from sklearn.ensemble import BaggingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier, StackingClassifier, AdaBoostClassifier

ENSEMBLE_LEARNING = ["Aucun", "Bagging", "Stacking", "Boosting"]
ENSEMBLE_LEARNING_NB_ESTIMATORS = 50

LOCAL_DATASET_NAMES = ["Iris (Classification)", "Penguins (Classification)", "Prix des voitures (Régression)"]
LOCAL_PATH_DATASET = ['Datasets/iris.csv', 'Datasets/penguins.csv', 'Datasets/CarPrice.csv']
PYDATASET_NAMES = ["cars", "quakes"]
PYDATASET_DISPLAY_NAMES = ["Speed and Stopping Distances of Cars (Régression)", "Locations of Earthquakes off Fiji (Data vizualisation)"]

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


def streamlit_code_regression(features_list_from_session, target_from_session, model=None, *, polynomial=False,
                              model1=None, model2=None):
    model_section = f"""
# modèle
model = {repr(model)}
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Métrique train set
MSE_reg_train = mean_squared_error(y_train, pred_train)
RMSE_reg_train = np.sqrt(MSE_reg_train)
MAE_reg_train = mean_absolute_error(y_train, pred_train)
r2_reg_train = r2_score(y_train, pred_train)

# Métrique test set
MSE_reg_test = mean_squared_error(y_test, pred_test)
RMSE_reg_test = np.sqrt(MSE_reg_test)
MAE_reg_test = mean_absolute_error(y_test, pred_test)
r2_reg_test = r2_score(y_test, pred_test)
""" if not polynomial else f"""
# modèles
model1 = {repr(model1)}
x_poly = model1.fit_transform(X_train)
model2 = {repr(model2)}
model2.fit(x_poly, y_train)
y_poly_pred_train = model2.predict(x_poly)
y_poly_pred_test = model2.predict(model1.fit_transform(X_test))

# Métrique train set
MSE_reg_train = mean_squared_error(y_train, y_poly_pred_train)
RMSE_reg_train = np.sqrt(MSE_reg_train)
MAE_reg_train = mean_absolute_error(y_train, y_poly_pred_train)
r2_reg_train = r2_score(y_train, y_poly_pred_train)
# Métrique test set
MSE_reg_test = mean_squared_error(y_test, y_poly_pred_test)
RMSE_reg_test = np.sqrt(MSE_reg_test)
MAE_reg_test = mean_absolute_error(y_test, y_poly_pred_test)
r2_reg_test = r2_score(y_test, y_poly_pred_test)
"""

    code = f"""
# df contient les données initiales avec pandas

# On nettoie le dataset
df_sans_NaN = pd.concat([df[{features_list_from_session}].reset_index(drop=True),
                        df['{target_from_session}'].reset_index(drop=True)],
                        axis=1).dropna()
X_train, X_test, y_train, y_test = train_test_split(
                    df_sans_NaN[{features_list_from_session}].values,
                    df_sans_NaN['{target_from_session}'], test_size=0.4, random_state=4)
X_train, X_test, y_train, y_test = scale(X_train), scale(X_test), scale(y_train), scale(y_test)
""" + model_section + """
# Learning curves
N, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
fig = go.Figure()
fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train', marker=dict(color='deepskyblue'))
fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation', marker=dict(color='red'))
fig.update_xaxes(title_text="Données de validation")
fig.update_yaxes(title_text="Score")
fig.update_layout(
    template='simple_white',
    font=dict(size=10),
    autosize=False,
    width=900, height=450,
    margin=dict(l=40, r=40, b=40, t=40),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    title={'text': "<b>Learning curves</b>",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
            }
)
plot(fig)
    """
    return code


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
    return distances[voisins - 1]


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
