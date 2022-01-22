# Importations
import itertools
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import binascii
import numpy as np
import pandas as pd
import time
import os
import webbrowser
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LinearRegression, PoissonRegressor, ElasticNet, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn import metrics
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import umap.umap_ as UMAP
from scipy.spatial import distance
from utils import *
import more_itertools

####### html/css config ########
st.set_page_config(layout="wide", page_title="No code AI", menu_items={
    'About': "No-code AI Platform - réalisé par Antonin"
})

st.markdown("""
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
    font-size:30px !important;
    font-weight: bold;
    text-align: center;
    text-decoration: underline;
    text-decoration-color: #4976E4;
    text-decoration-thickness: 5px;
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
""", unsafe_allow_html=True)
st.sidebar.image("logo/NAP_logo.png", use_column_width=True)
st.sidebar.write("##")


###### Load data #######
def load_data():
    try:
        if 'csv' in st.session_state.file_details['FileName']:
            if st.session_state.separateur != "":
                st.session_state.data = pd.read_csv(uploaded_file, sep=st.session_state.separateur, engine='python')
            else:
                st.session_state.data = pd.read_csv(uploaded_file)
        else:
            if st.session_state.separateur != "":
                st.session_state.data = pd.read_excel(uploaded_file, sep=st.session_state.separateur, engine='python')
            else:
                st.session_state.data = pd.read_excel(uploaded_file)
    except:
        pass


##################################
####### Code streamlit app #######
##################################

# Session
if "col_to_time" not in st.session_state:
    st.session_state.col_to_time = ""
if "drop_col" not in st.session_state:
    st.session_state.drop_col = ""
if "col_to_float_money" not in st.session_state:
    st.session_state.col_to_float_money = ""
if "col_to_float_coma" not in st.session_state:
    st.session_state.col_to_float_coma = ""
if "separateur" not in st.session_state:
    st.session_state.separateur = ""
if "slider_col" not in st.session_state:
    st.session_state.slider_col = ""
if "degres" not in st.session_state:
    st.session_state.degres = ""
if "file_details" not in st.session_state:
    st.session_state.file_details = ""

# Pages principales
PAGES = ["Accueil", "Dataset", "Analyse des colonnes", "Matrice de corrélations", "Section graphiques",
         "Régressions", "Classification", "Ensemble learning", "Réduction de dimension"]
st.sidebar.title('Menu :bulb:')
choix_page = st.sidebar.radio(label="", options=PAGES)


