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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LinearRegression, PoissonRegressor, ElasticNet, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn import metrics
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from umap import UMAP
from scipy.spatial import distance
from utils import *
import more_itertools

####### html/css config ########
st.set_page_config(layout="wide")
st.markdown("""
<style>
.first_titre {
    font-size:75px !important;
    font-weight: bold;
    box-sizing: border-box;
    text-align: center;
    width: 100%;
    // border: solid #4976E4 5px;
    // padding: 5px;
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
.grand_titre_section_ML_DL {
    font-size:40px !important;
    font-weight: bold;
    text-decoration: underline;
    text-decoration-color: #2782CD;
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
.caract{
    font-size:11px !important;
}
.nom_colonne_page3{
    font-size:17px !important;
    text-decoration: underline;
    text-decoration-color: #000;
    text-decoration-thickness: 1px;
}
.center{
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
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
         "Machine Learning"]
st.sidebar.title('Menu :bulb:')
choix_page = st.sidebar.radio(label="", options=PAGES)

############# Page 1 #############
if choix_page == "Accueil":
    st.markdown('<p class="first_titre">No-code AI Platform</p>', unsafe_allow_html=True)
    st.write("---")
    c1, c2 = st.columns((3, 2))
    with c2:
        st.write("##")
        st.write("##")
        st.image("logo/background.png")
    st.write("##")
    with c1:
        st.write("##")
        st.markdown(
            '<p class="intro">Bienvenue sur la <b>no-code AI platform</b> ! Déposez vos datasets csv ou excel ou choisissez en un parmi ceux proposés et commencez votre analyse dès maintenant ! Cherchez les variables les plus intéressantes, visualisez vos données, et créez vos modèles de Machine Learning en toute simplicité.' +
            ' Si vous choisissez de travailler avec votre dataset et que vous voulez effectuez des modifications sur celui-ci, il faudra le télécharger une fois les modifications faites pour pouvoir l\'utiliser sur les autres pages. </p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="intro">Un tutoriel sur l\'utilisation de ce site est disponible sur le repo Github. En cas de bug ou d\'erreur veuillez m\'en informer par mail ou sur Discord.</p>',
            unsafe_allow_html=True)
        st.markdown(
            '<p class="intro"><b>Commencez par choisir un dataset dans la section Dataset !</b></p>',
            unsafe_allow_html=True)
    c1, _, _, _, _, _ = st.columns(6)
    with c1:
        st.subheader("Liens")
        st.write(
            "• [Mon profil GitHub](https://github.com/antonin-lfv/Online_preprocessing_for_ML/blob/master/README.md)")
        st.write("• [Mon site](https://antonin-lfv.github.io)")
############# Page 1 #############

############# Page 2 #############
elif choix_page == 'Dataset':
    st.markdown('<p class="grand_titre">Chargement du dataset</p>', unsafe_allow_html=True)
    st.write('##')
    col1_1, b_1, col2_1 = st.columns((1, 0.1, 1))
    col1, b, col2 = st.columns((2.7, 0.2, 1))
    with col1_1:
        dataset_choix = st.selectbox("Dataset",
                                     ["-- Choisissez une option --", "Iris (Classification)",
                                      "Penguins (Classification)",
                                      "Choisir un dataset personnel"], )
        message_ = st.empty()

    if 'choix_dataset' in st.session_state:
        with col1_1:
            message_.success(st.session_state.choix_dataset)

    if dataset_choix == "-- Choisissez une option --":
        if 'choix_dataset' in st.session_state:
            if st.session_state.choix_dataset == "Le fichier chargé est le dataset des iris" or st.session_state.choix_dataset == "Le fichier chargé est le dataset des penguins":
                with col1:
                    st.write("##")
                    st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                    st.write(st.session_state.data.head(50))
                    st.write("##")

                with col2:
                    st.write("##")
                    st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                    st.write(' - Taille:', st.session_state.data.shape)
                    st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                    st.write(' - Type des colonnes:', st.session_state.data.dtypes.value_counts())
                    st.write(' - Pourcentage de valeurs manquantes:', round(
                        sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                                st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                             ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')
            elif st.session_state.choix_dataset == "Vous avez choisi de selectionner votre dataset" and 'data' in st.session_state:
                with col1:
                    st.write("##")
                    st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                    st.write(st.session_state.data.head(50))
                    st.write("##")

                with col2:
                    st.write("##")
                    st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                    st.write(' - Taille:', st.session_state.data.shape)
                    st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                    st.write(' - Type des colonnes:', st.session_state.data.dtypes.value_counts())
                    st.write(' - Pourcentage de valeurs manquantes:', round(
                        sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                                st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                             ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')

                col1_modif, col2_modif = st.columns((2, 1))
                with col1_modif:
                    st.write("##")
                    st.info("Rechargez votre dataset pour le modifier, en cliquant 2 FOIS sur le bouton ci-dessous")
                if st.button("Modifier le dataset"):
                    dataset_choix = ""
                    st.session_state.choix_dataset = ""
                    st.session_state.clear()

    if dataset_choix == "Iris (Classification)":
        st.session_state.data = pd.read_csv('Datasets/iris.csv')
        st.session_state.choix_dataset = "Le fichier chargé est le dataset des iris"
        with col1_1:
            message_.success(st.session_state.choix_dataset)

        with col1:
            st.write("##")
            st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
            st.write(st.session_state.data.head(50))
            st.write("##")

        with col2:
            st.write("##")
            st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
            st.write(' - Taille:', st.session_state.data.shape)
            st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
            st.write(' - Type des colonnes:', st.session_state.data.dtypes.value_counts())
            st.write(' - Pourcentage de valeurs manquantes:', round(
                sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                        st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                     ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')

    if dataset_choix == "Penguins (Classification)":
        st.session_state.data = pd.read_csv('Datasets/penguins.csv')
        st.session_state.choix_dataset = "Le fichier chargé est le dataset des penguins"
        with col1_1:
            message_.success(st.session_state.choix_dataset)

        with col1:
            st.write("##")
            st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
            st.write(st.session_state.data.head(50))
            st.write("##")

        with col2:
            st.write("##")
            st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
            st.write(' - Taille:', st.session_state.data.shape)
            st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
            st.write(' - Type des colonnes:', st.session_state.data.dtypes.value_counts())
            st.write(' - Pourcentage de valeurs manquantes:', round(
                sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                        st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                     ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')


    elif dataset_choix == "Choisir un dataset personnel":

        with col1_1:
            uploaded_file = st.file_uploader("", type=['csv', 'xls'])
            # uploaded_file = 0
            if uploaded_file is not None:
                st.session_state.file_details = {"FileName": uploaded_file.name,
                                                 "FileType": uploaded_file.type,
                                                 "FileSize": uploaded_file.size}
                st.success('Fichier ' + st.session_state.file_details['FileName'] + ' chargé avec succès !')

        if 'data' in st.session_state:
            del st.session_state.data

        if uploaded_file == None:
            with col1_1:
                st.info("Veuillez charger un dataset")

        if "data" not in st.session_state:
            load_data()

        if "data" in st.session_state:
            my_expander = st.expander(label="Options de preprocessing")
            with my_expander:
                with col1_1:
                    st.write("##")
                    st.write("##")
                    st.write("##")
                    st.session_state.separateur = st.text_input("Séparateur (optionnel): ")
                st.write("##")

                load_data()

                st.markdown("<p class='petite_section'>Modifications du dataset : </p>", unsafe_allow_html=True)
                col1_1, b_1, col2_1, c_1, col3_1 = st.columns((1, 0.2, 1, 0.2, 1))  # pour time series
                st.write("##")
                option_col_update = st.session_state.data.columns.tolist()

                with col1_1:
                    st.session_state.col_to_time = st.multiselect(label='Conversion Time Series',
                                                                  options=option_col_update,
                                                                  )
                with col2_1:
                    st.session_state.col_to_float_money = st.multiselect('Conversion Monnaies',
                                                                         options=option_col_update,
                                                                         )
                with col3_1:
                    st.session_state.col_to_float_coma = st.multiselect('Conversion string avec virgules vers float',
                                                                        options=option_col_update,
                                                                        )
                with col1_1:
                    st.session_state.drop_col = st.multiselect(label='Drop columns',
                                                               options=option_col_update,
                                                               )

                with col1_1:
                    for col in st.session_state["col_to_time"]:
                        try:
                            st.session_state.data[col] = pd.to_datetime(st.session_state.data[col])
                            st.success("Transformation de " + col + " effectuée !")
                        except:
                            st.error("Transformation impossible ou déjà effectuée")
                with col2_1:
                    for col in st.session_state.col_to_float_money:
                        try:
                            st.session_state.data[col] = st.session_state.data[col].apply(clean_data).astype('float')
                            st.success("Transformation de " + col + " effectuée !")
                        except:
                            st.error("Transformation impossible ou déjà effectuée")
                with col3_1:
                    for col in st.session_state.col_to_float_coma:
                        try:
                            st.session_state.data[col] = st.session_state.data[col].apply(
                                lambda x: float(str(x).replace(',', '.')))
                            st.success("Transformation de " + col + " effectuée !")
                        except:
                            st.error("Transformation impossible ou déjà effectuée")
                with col1_1:
                    for col in st.session_state["drop_col"]:
                        try:
                            st.session_state.data = st.session_state.data.drop(columns=col, axis=1)
                            st.success("Colonnes " + col + " supprimée !")
                        except:
                            st.error("Transformation impossible ou déjà effectuée")

                with col1:
                    st.write("##")
                    st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                    st.write(st.session_state.data.head(50))
                    st.write("##")

                with col2:
                    st.write("##")
                    st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                    st.write(' - Taille:', st.session_state.data.shape)
                    st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                    st.write(' - Type des colonnes:', st.session_state.data.dtypes.value_counts())
                    st.write(' - Pourcentage de valeurs manquantes:', round(
                        sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                                st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                             ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')
                st.download_button(data=st.session_state.data.to_csv(), label="Télécharger le dataset modifié",
                                   file_name='dataset.csv')
        st.session_state.choix_dataset = "Vous avez choisi de selectionner votre dataset"
        with col1_1:
            message_.success(st.session_state.choix_dataset)
############# Page 2 #############

############# Page 3 #############
elif choix_page == "Analyse des colonnes":
    st.markdown('<p class="grand_titre">Analyse des colonnes</p>', unsafe_allow_html=True)
    st.write('##')
    if 'data' in st.session_state:
        options = st.session_state.data.columns.to_list()
        st.session_state.slider_col = st.multiselect(
            'Selectionner une ou plusieurs colonnes',
            options, help="Choisissez les colonnes à analyser"
        )
        if st.session_state.slider_col:
            col1, b, col2, c = st.columns((1.1, 0.1, 1.1, 0.3))
            with col1:
                st.write('##')
                st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
            with col2:
                st.write('##')
                st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
            for col in st.session_state.slider_col:
                ### Données ###
                data_col = st.session_state.data[col].copy()
                n_data = st.session_state.data[col].to_numpy()

                st.write('##')
                col1, b, col2, c = st.columns((1, 1, 2, 0.5))
                with col1:
                    st.markdown('<p class="nom_colonne_page3">' + col + '</p>', unsafe_allow_html=True)
                    st.write(data_col.head(20))
                with col2:
                    st.write('##')
                    st.write(' ● type de la colonne :', type(data_col))
                    st.write(' ● type des valeurs :', type(data_col.iloc[1]))
                    if n_data.dtype == float:
                        moyenne = data_col.mean()
                        variance = data_col.std()
                        max = data_col.max()
                        min = data_col.min()
                        st.write(' ● Moyenne :', round(moyenne, 3))

                        st.write(' ● Variance :', round(variance, 3))

                        st.write(' ● Maximum :', max)

                        st.write(' ● Minimum :', min)

                    st.write(' ● Valeurs les plus présentes:', (Counter(n_data).most_common()[0])[0], 'apparait',
                             (Counter(n_data).most_common()[0])[1], 'fois', ', ', (Counter(n_data).most_common()[1])[0],
                             'apparait',
                             (Counter(n_data).most_common()[1])[1], 'fois')

                    st.write(' ● Nombre de valeurs manquantes:',
                             sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))

                    st.write(' ● Longueur:', n_data.shape[0])

                    st.write(' ● Nombre de valeurs différentes non NaN:',
                             abs(len(Counter(n_data)) - sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
                    ### Fin section données ###
                st.write('##')

    else:
        st.info("Veuillez charger vos données dans la section Dataset")
############# Page 3 #############

############# Page 4 #############
elif choix_page == "Matrice de corrélations":
    st.markdown('<p class="grand_titre">Matrice de corrélations</p>', unsafe_allow_html=True)
    st.write("##")
    if 'data' in st.session_state:
        col1, b, col2 = st.columns((1, 1, 2))
        df_sans_NaN = st.session_state.data
        with col1:
            st.session_state.couleur_corr = st.selectbox('Couleur',
                                                         [
                                                             '-- Selectionner une colonne --'] + df_sans_NaN.columns.tolist(),
                                                         help="Choisissez la variable catégorielle pour la coloration des classes"
                                                         )
            st.write("##")
        st.session_state.select_columns_corr = st.multiselect("Choisir au moins deux colonnes",
                                                              ["Toutes les colonnes"] + col_numeric(df_sans_NaN),
                                                              help="Choisissez vos features"
                                                              )
        if len(st.session_state.select_columns_corr) > 1 and "Toutes les colonnes" not in st.session_state.select_columns_corr:
            df_sans_NaN = pd.concat([st.session_state.data[col] for col in st.session_state.select_columns_corr],
                                    axis=1).dropna()
            if len(df_sans_NaN) == 0:
                st.write("##")
                st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                if st.session_state.couleur_corr != '-- Selectionner une colonne --':
                    fig = px.scatter_matrix(st.session_state.data,
                                            dimensions=col_numeric(df_sans_NaN[st.session_state.select_columns_corr]),
                                            color=st.session_state.couleur_corr, color_continuous_scale='Bluered_r')
                else:
                    fig = px.scatter_matrix(df_sans_NaN,
                                            dimensions=col_numeric(df_sans_NaN[st.session_state.select_columns_corr]))
                fig.update_layout(width=1000, height=700, margin=dict(l=40, r=50, b=40, t=40), font=dict(size=10))
                fig.update_layout({"xaxis" + str(i + 1): dict(showticklabels=False) for i in
                                   range(len(col_numeric(df_sans_NaN[st.session_state.select_columns_corr])))})
                fig.update_layout({"yaxis" + str(i + 1): dict(showticklabels=False) for i in
                                   range(len(col_numeric(df_sans_NaN[st.session_state.select_columns_corr])))})
                fig.update_traces(marker=dict(size=7))
                fig.update_traces(diagonal_visible=False)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig)
        elif st.session_state.select_columns_corr == ["Toutes les colonnes"]:
            df_sans_NaN = st.session_state.data.dropna()
            if len(df_sans_NaN) == 0:
                st.write("##")
                st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                if st.session_state.couleur_corr != 'Selectionner une colonne':
                    fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN),
                                            color=st.session_state.couleur_corr)
                else:
                    fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN))
                fig.update_layout(
                    {"xaxis" + str(i + 1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
                fig.update_layout(
                    {"yaxis" + str(i + 1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
                fig.update_traces(marker=dict(size=2))
                fig.update_layout(width=1000, height=700, margin=dict(l=40, r=50, b=40, t=40), font=dict(size=10))
                fig.update_traces(marker=dict(size=7))
                fig.update_traces(diagonal_visible=False)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig)
        elif len(
                st.session_state.select_columns_corr) > 1 and "Toutes les colonnes" in st.session_state.select_columns_corr:
            st.error("Erreur de saisi !")
        else:
            pass
    else:
        st.info("Veuillez charger vos données dans la section Dataset")
############# Page 4 #############

############# Page 5 #############
elif choix_page == "Section graphiques":
    st.markdown('<p class="grand_titre">Graphiques et regressions</p>', unsafe_allow_html=True)
    st.write("##")
    if 'data' in st.session_state:
        col1, col2, col3, col4 = st.columns(4)  # pour les autres select
        col_num = col_numeric(st.session_state.data) + col_temporal(st.session_state.data)
        with col1:
            with st.expander("Données"):
                st.session_state.abscisse_plot = st.selectbox('Données en abscisses', col_num,
                                                              )
                st.session_state.ordonnee_plot = st.selectbox('Données en ordonnées', col_num[::-1],
                                                              )
        with col2:
            with st.expander("Type de graphique"):
                st.session_state.type_plot = st.radio(label="",
                                                      options=['Points', 'Courbe', 'Latitude/Longitude', 'Histogramme'],
                                                      help="Choisissez le type qui vous convient"
                                                      )
                type_plot_dict = {
                    'Courbe': 'lines',
                    'Points': 'markers',
                    'Latitude/Longitude': 'map',
                }
        if st.session_state.abscisse_plot and st.session_state.ordonnee_plot:
            if st.session_state.type_plot == 'Latitude/Longitude':
                fig = go.Figure()
                df_sans_NaN = pd.concat([st.session_state.data[st.session_state.abscisse_plot].reset_index(drop=True),
                                         st.session_state.data[st.session_state.ordonnee_plot].reset_index(drop=True)],
                                        axis=1).dropna()
                if len(df_sans_NaN) == 0:
                    st.warning('Le dataset composé des 2 colonnes selectionnées après dropna() est vide !')
                else:
                    fig.add_scattermapbox(
                        mode="markers",
                        lon=df_sans_NaN[st.session_state.ordonnee_plot],
                        lat=df_sans_NaN[st.session_state.abscisse_plot],
                        marker={'size': 10,
                                'color': 'firebrick',
                                })
                    fig.update_layout(
                        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                        mapbox={
                            'center': {'lon': -80, 'lat': 40},
                            'style': "stamen-terrain",
                            'zoom': 1})
                    st.plotly_chart(fig)
            elif st.session_state.type_plot == 'Histogramme':
                fig = go.Figure()
                df_sans_NaN = pd.concat([st.session_state.data[st.session_state.abscisse_plot].reset_index(drop=True),
                                         st.session_state.data[st.session_state.ordonnee_plot].reset_index(drop=True)],
                                        axis=1).dropna()
                if len(df_sans_NaN) == 0:
                    st.warning('Le dataset composé des 2 colonnes selectionnées après dropna() est vide !')
                else:
                    fig.add_histogram(x=df_sans_NaN[st.session_state.abscisse_plot],
                                      y=df_sans_NaN[st.session_state.ordonnee_plot])
            else:
                with col3:
                    with st.expander("Indices statistiques"):
                        st.checkbox("Maximum", key="maximum")
                        st.session_state.moyenne = st.checkbox("Moyenne")
                        st.session_state.minimum = st.checkbox("Minimum")
                fig = go.Figure()
                df_sans_NaN = pd.concat([st.session_state.data[st.session_state.abscisse_plot].reset_index(drop=True),
                                         st.session_state.data[st.session_state.ordonnee_plot].reset_index(drop=True)],
                                        axis=1).dropna()
                if len(df_sans_NaN) == 0:
                    st.warning('Le dataset composé des 2 colonnes selectionnées après dropna() est vide !')
                else:
                    fig.add_scatter(x=df_sans_NaN[st.session_state.abscisse_plot],
                                    y=df_sans_NaN[st.session_state.ordonnee_plot],
                                    mode=type_plot_dict[st.session_state.type_plot], name='', showlegend=False)
                    # if abscisse_plot not in col_to_time and ordonnee_plot not in col_to_time :
                    with col4:
                        if st.session_state.type_plot == 'Points' or st.session_state.type_plot == 'Courbe':
                            if st.session_state.abscisse_plot not in st.session_state.col_to_time and st.session_state.ordonnee_plot not in st.session_state.col_to_time:
                                with st.expander("Regréssion"):
                                    st.session_state.trendline = st.checkbox("Regression linéaire")
                                    st.session_state.polynom_feat = st.checkbox("Regression polynomiale")
                                    if st.session_state.polynom_feat:
                                        st.session_state.degres = st.slider('Degres de la regression polynomiale',
                                                                            min_value=2,
                                                                            max_value=100)
                    equation_col_1, equation_col_2 = st.columns(2)
                    if st.session_state.trendline:
                        # regression linaire
                        X = df_sans_NaN[st.session_state.abscisse_plot].values.reshape(-1, 1)
                        model = LinearRegression()
                        model.fit(X, df_sans_NaN[st.session_state.ordonnee_plot])
                        x_range = np.linspace(X.min(), X.max(), len(df_sans_NaN[st.session_state.ordonnee_plot]))
                        y_range = model.predict(x_range.reshape(-1, 1))
                        fig.add_scatter(x=x_range, y=y_range, name='<b>Regression linéaire<b>', mode='lines',
                                        marker=dict(color='red'))
                        with equation_col_1:
                            with st.expander('Équation Linear regression'):
                                st.write(f'**f(x) = {model.coef_[0].round(3)}x + {model.intercept_.round(2)}**')
                        # #################
                    if st.session_state.polynom_feat:
                        # regression polynomiale
                        X = df_sans_NaN[st.session_state.abscisse_plot].values.reshape(-1, 1)
                        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                        poly = PolynomialFeatures(st.session_state.degres)
                        poly.fit(X)
                        X_poly = poly.transform(X)
                        x_range_poly = poly.transform(x_range)
                        model = LinearRegression(fit_intercept=False)
                        model.fit(X_poly, df_sans_NaN[st.session_state.ordonnee_plot])
                        y_poly = model.predict(x_range_poly)

                        eq_poly = model.coef_.tolist()
                        eq_poly[0] += model.intercept_
                        puissances_x = poly.get_feature_names_out()
                        eq_final = ""
                        for i, j in zip(eq_poly, puissances_x):
                            eq_final += f'{round(i, 2)}{j} + '
                        fig.add_scatter(x=x_range.squeeze(), y=y_poly, name='<b>Polynomial Features<b>',
                                        marker=dict(color='green'))
                        with equation_col_2:
                            with st.expander('Équation polynomial features'):
                                st.write(f'**f(x0) = {eq_final[:-2]}**')
                        # #################

                    if st.session_state.moyenne:
                        # Moyenne #
                        fig.add_hline(y=df_sans_NaN[st.session_state.ordonnee_plot].mean(),
                                      line_dash="dot",
                                      annotation_text="moyenne : {}".format(
                                          round(df_sans_NaN[st.session_state.ordonnee_plot].mean(), 1)),
                                      annotation_position="bottom left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
                        pass
                    if st.session_state.minimum:
                        # Minimum #
                        fig.add_hline(y=df_sans_NaN[st.session_state.ordonnee_plot].min(),
                                      line_dash="dot",
                                      annotation_text="minimum : {}".format(
                                          round(df_sans_NaN[st.session_state.ordonnee_plot].min(), 1)),
                                      annotation_position="bottom left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
                        pass
                    if st.session_state.maximum:
                        # Maximum #
                        fig.add_hline(y=df_sans_NaN[st.session_state.ordonnee_plot].max(),
                                      line_dash="dot",
                                      annotation_text="maximum : {}".format(
                                          round(df_sans_NaN[st.session_state.ordonnee_plot].max(), 1)),
                                      annotation_position="top left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
                        pass
            if len(df_sans_NaN) != 0:
                fig.update_xaxes(title_text=st.session_state.abscisse_plot)
                fig.update_yaxes(title_text=st.session_state.ordonnee_plot)
                fig.update_layout(
                    template='simple_white',
                    font=dict(size=10),
                    autosize=False,
                    width=1300, height=650,
                    margin=dict(l=40, r=50, b=40, t=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.write("##")
                st.plotly_chart(fig)
    else:
        st.info("Veuillez charger vos données dans la section Dataset")
############# Page 5 #############

############# ML section #############
elif choix_page == "Machine Learning":
    # Pages
    PAGES_ML = ["Regression", "K-nearest neighbors", "K-Means", "Support Vector Machine", "PCA", "UMAP"]
    st.sidebar.title('Machine Learning  :brain:')
    st.sidebar.radio(label="", options=PAGES_ML, key="choix_page_ml")

    if st.session_state.choix_page_ml == "Regression":
        st.markdown('<p class="grand_titre">Régression</p>', unsafe_allow_html=True)
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.5, 1, 0.5))
        with exp2:
            with st.expander("Les principaux algorithmes de regression"):
                st.write("""
                * Régression linéaire
                * Régression polynomiale
                * Régression de poisson
                * Elastic Net
                * Ridge
                * Lasso
                """)
        if 'data' in st.session_state:
            _, col1_abscisse_reg, col1_ordonnee_reg, _ = st.columns((0.1, 0.5, 0.5, 0.1))
            _, warning1, _ = st.columns((0.1, 1, 0.1))
            _, box1_title, _ = st.columns((0.1, 1, 0.1))
            _, box1_eval1, box1_eval2, box1_eval3, box1_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
            _, box1, _ = st.columns((0.1, 1, 0.1))
            _, box2_title, _ = st.columns((0.1, 1, 0.1))
            _, box2_eval1, box2_eval2, box2_eval3, box2_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
            _, box2, _ = st.columns((0.1, 1, 0.1))
            _, box3_title, _ = st.columns((0.1, 1, 0.1))
            _, box3_eval1, box3_eval2, box3_eval3, box3_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
            _, box3, _ = st.columns((0.1, 1, 0.1))
            _, box4_title, _ = st.columns((0.1, 1, 0.1))
            _, box4_eval1, box4_eval2, box4_eval3, box4_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
            _, box4, _ = st.columns((0.1, 1, 0.1))
            _, box5_title, _ = st.columns((0.1, 1, 0.1))
            _, box5_eval1, box5_eval2, box5_eval3, box5_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
            _, box5, _ = st.columns((0.1, 1, 0.1))
            _, box6_title, _ = st.columns((0.1, 1, 0.1))
            _, box6_eval1, box6_eval2, box6_eval3, box6_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
            _, box6, _ = st.columns((0.1, 1, 0.1))
            with exp2:
                st.write("##")
                st.markdown('<p class="section">Selection des features pour la régression</p>', unsafe_allow_html=True)
            with col1_abscisse_reg:
                st.session_state.choix_features_reg = st.multiselect("Features",
                                                                     col_numeric(st.session_state.data),
                                                                     help="Choisissez au moins 1 features"
                                                                     )
            with col1_ordonnee_reg:
                st.session_state.choix_target_reg = st.selectbox("Target",
                                                                 col_numeric(st.session_state.data)[::-1],
                                                                 )
                st.write("##")
            if st.session_state.choix_features_reg == st.session_state.choix_target_reg:
                with warning1:
                    st.warning("Vous ne pouvez pas choisir deux fois la même feature")
            else:
                # Chargement des données - On enlève les valeurs manquantes
                df_sans_NaN = pd.concat(
                    [st.session_state.data[st.session_state.choix_features_reg].reset_index(drop=True),
                     st.session_state.data[st.session_state.choix_target_reg].reset_index(drop=True)],
                    axis=1).dropna()
                if len(df_sans_NaN) == 0:
                    with exp2:
                        st.warning('Le dataset composé des 2 colonnes selectionnées après dropna() est vide !')

                elif len(st.session_state.choix_features_reg) > 0:  # Le dataset est viable
                    df_sans_NaN
                    try:
                        # Data
                        X_train, X_test, y_train, y_test = train_test_split(
                            st.session_state.data[st.session_state.choix_features_reg].values.reshape(-1, 1),
                            st.session_state.data[st.session_state.choix_target_reg], test_size=0.4, random_state=4)
                        # ###############################################################################
                        with box1_title:
                            st.write("##")
                            st.write('---')
                            st.write("##")
                            st.write("##")
                            st.markdown('<p class="section">Régression linéaire</p>', unsafe_allow_html=True)
                            st.write("##")
                        # Modèle
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        pred_train = model.predict(X_train)
                        pred_test = model.predict(X_test.reshape(-1, 1))
                        x_plot_line_reg = X_train
                        y_plot_line_reg = pred_test
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
                        # Affichage métriques
                        with box1_eval1:
                            st.metric(label="MSE (par rapport au train)", value=round(MSE_reg_test, 3),
                                      delta=round(MSE_reg_test - MSE_reg_train, 3))
                        with box1_eval2:
                            st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                      delta=round(RMSE_reg_test - RMSE_reg_train, 3))
                        with box1_eval3:
                            st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                      delta=round(MAE_reg_test - MAE_reg_train, 3))
                        with box1_eval4:
                            st.metric(label="r² (par rapport au train)", value=round(r2_reg_test, 3),
                                      delta=round(r2_reg_test - r2_reg_train, 3))
                        # Learning curves
                        N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
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
                        with box1:
                            st.write("##")
                            st.plotly_chart(fig)
                            st.write('---')
                            st.write("##")

                        # ###############################################################################
                        with box2_title:
                            st.write("##")
                            st.markdown('<p class="section">Régression polynomiale</p>', unsafe_allow_html=True)
                            st.write("##")
                        # Modèle
                        model1 = PolynomialFeatures(degree=4)
                        x_poly = model1.fit_transform(X_train)
                        model2 = LinearRegression(fit_intercept=False)
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
                        # Affichage métriques
                        with box2_eval1:
                            st.metric(label="MSE (par rapport au train)", value=round(MSE_reg_test, 3),
                                      delta=round(MSE_reg_test - MSE_reg_train, 3))
                        with box2_eval2:
                            st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                      delta=round(RMSE_reg_test - RMSE_reg_train, 3))
                        with box2_eval3:
                            st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                      delta=round(MAE_reg_test - MAE_reg_train, 3))
                        with box2_eval4:
                            st.metric(label="r² (par rapport au train)", value=round(r2_reg_test, 3),
                                      delta=round(r2_reg_test - r2_reg_train, 3))
                        # Learning curves
                        N, train_score, val_score = learning_curve(model2, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
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
                        with box2:
                            st.write("##")
                            st.plotly_chart(fig)
                            st.write('---')
                            st.write("##")

                        # ###############################################################################
                        with box3_title:
                            st.write("##")
                            st.markdown('<p class="section">Régression de poisson</p>', unsafe_allow_html=True)
                            st.write("##")
                        # Modèle
                        model = PoissonRegressor()
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
                        # Affichage métriques
                        with box3_eval1:
                            st.metric(label="MSE (par rapport au train)", value=round(MSE_reg_test, 3),
                                      delta=round(MSE_reg_test - MSE_reg_train, 3))
                        with box3_eval2:
                            st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                      delta=round(RMSE_reg_test - RMSE_reg_train, 3))
                        with box3_eval3:
                            st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                      delta=round(MAE_reg_test - MAE_reg_train, 3))
                        with box3_eval4:
                            st.metric(label="r² (par rapport au train)", value=round(r2_reg_test, 3),
                                      delta=round(r2_reg_test - r2_reg_train, 3))
                        # Learning curves
                        N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
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
                        with box3:
                            st.write("##")
                            st.plotly_chart(fig)
                            st.write('---')
                            st.write("##")

                        # ###############################################################################
                        with box4_title:
                            st.write("##")
                            st.markdown('<p class="section">Elastic net</p>', unsafe_allow_html=True)
                            st.write("##")
                        # Modèle
                        model = ElasticNet()
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
                        # Affichage métriques
                        with box4_eval1:
                            st.metric(label="MSE (par rapport au train)", value=round(MSE_reg_test, 3),
                                      delta=round(MSE_reg_test - MSE_reg_train, 3))
                        with box4_eval2:
                            st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                      delta=round(RMSE_reg_test - RMSE_reg_train, 3))
                        with box4_eval3:
                            st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                      delta=round(MAE_reg_test - MAE_reg_train, 3))
                        with box4_eval4:
                            st.metric(label="r² (par rapport au train)", value=round(r2_reg_test, 3),
                                      delta=round(r2_reg_test - r2_reg_train, 3))
                        # Learning curves
                        N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
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
                        with box4:
                            st.write("##")
                            st.plotly_chart(fig)
                            st.write('---')
                            st.write("##")

                        # ###############################################################################
                        with box5_title:
                            st.write("##")
                            st.markdown('<p class="section">Ridge</p>', unsafe_allow_html=True)
                            st.write("##")
                        # Modèle
                        model = Ridge()
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
                        # Affichage métriques
                        with box5_eval1:
                            st.metric(label="MSE (par rapport au train)", value=round(MSE_reg_test, 3),
                                      delta=round(MSE_reg_test - MSE_reg_train, 3))
                        with box5_eval2:
                            st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                      delta=round(RMSE_reg_test - RMSE_reg_train, 3))
                        with box5_eval3:
                            st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                      delta=round(MAE_reg_test - MAE_reg_train, 3))
                        with box5_eval4:
                            st.metric(label="r² (par rapport au train)", value=round(r2_reg_test, 3),
                                      delta=round(r2_reg_test - r2_reg_train, 3))
                        # Learning curves
                        N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
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
                        with box5:
                            st.write("##")
                            st.plotly_chart(fig)
                            st.write('---')
                            st.write("##")

                        # ###############################################################################
                        with box6_title:
                            st.write("##")
                            st.markdown('<p class="section">Lasso</p>', unsafe_allow_html=True)
                            st.write("##")
                        # Modèle
                        model = Lasso()
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
                        # Affichage métriques
                        with box6_eval1:
                            st.metric(label="MSE (par rapport au train)", value=round(MSE_reg_test, 3),
                                      delta=round(MSE_reg_test - MSE_reg_train, 3))
                        with box6_eval2:
                            st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                      delta=round(RMSE_reg_test - RMSE_reg_train, 3))
                        with box6_eval3:
                            st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                      delta=round(MAE_reg_test - MAE_reg_train, 3))
                        with box6_eval4:
                            st.metric(label="r² (par rapport au train)", value=round(r2_reg_test, 3),
                                      delta=round(r2_reg_test - r2_reg_train, 3))
                        # Learning curves
                        N, train_score, val_score = learning_curve(model, X_train, y_train,
                                                                   train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
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
                        with box6:
                            st.write("##")
                            st.plotly_chart(fig)
                            st.write('---')
                            st.write("##")

                    except:
                        with box2_title:
                            st.info("Impossible d'effectuer les régressions avec ces données")

        else:
            with exp2:
                st.write("##")
                st.info('Rendez-vous dans la section Dataset pour importer votre dataset')

    elif st.session_state.choix_page_ml == "K-nearest neighbors":
        st.markdown('<p class="grand_titre">KNN : k-nearest neighbors</p>', unsafe_allow_html=True)
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        with exp2:
            with st.expander("Principe de l'algorithme KNN"):
                st.write("""
                * 1ère étape : Choix du nombre de voisins k
                * 2ème étape : Calcul de la distance entre le point non classifié et tous les autre
                * 3ème étape : Sélection des k plus proches voisins
                * 4ème étape : On compte le nombre de voisins dans chaque classe
                * 5ème étape : Attribution de la classe la plus présente à notre point 
                """)
        if 'data' in st.session_state:
            _, col1_features_encode, _ = st.columns((0.1, 1, 0.1))
            _, sub_col1, _ = st.columns((0.4, 0.5, 0.4))
            _, col1_target, _ = st.columns((0.1, 1, 0.1))
            _, col_best_score, _ = st.columns((0.4, 0.4, 0.4))
            _, col_titre_eval_model, _ = st.columns((0.4, 0.4, 0.4))
            _, col1_eval_modele, col2_eval_modele, col3_eval_modele, col4_eval_modele, _ = st.columns(
                (0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
            _, col1_roc, _ = st.columns((0.1, 1, 0.1))
            _, col1_AUC_value, _ = st.columns((0.4, 0.4, 0.4))
            _, col_learning_curve, _ = st.columns((0.1, 4, 0.1))
            _, col_data_to_predict, _ = st.columns((0.1, 0.75, 0.1))
            _, col_pca_knn, _ = st.columns((0.1, 1, 0.1))
            _, sub_col_prediction_knn, _ = st.columns((0.4, 0.75, 0.4))
            _, col_pca_knn_plot, _ = st.columns((0.1, 4, 0.1))
            with col1_features_encode:
                st.write("##")
                st.markdown('<p class="section">Selection des colonnes pour le modèle (target+features)</p>',
                            unsafe_allow_html=True)
                st.session_state.choix_col = st.multiselect("Choisir au moins deux colonnes",
                                                            st.session_state.data.columns.tolist(),
                                                            )
            if len(st.session_state.choix_col) > 1:
                df_ml = st.session_state.data[st.session_state.choix_col]
                df_ml = df_ml.dropna(axis=0)
                if len(df_ml) == 0:
                    with col1_features_encode:
                        st.write("##")
                        st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else:
                    # encodage !
                    df_origine = df_ml.copy()
                    with col1_features_encode:
                        st.session_state.col_to_encodage = st.multiselect("Selectionner les colonnes à encoder",
                                                                          st.session_state.choix_col,
                                                                          )
                    with sub_col1:
                        with st.expander('Encodage'):
                            for col in st.session_state.col_to_encodage:
                                st.write("Colonne " + col + "  :  " + str(df_ml[col].unique().tolist()) + " -> " + str(
                                    np.arange(len(df_ml[col].unique()))))
                                df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())),
                                                   inplace=True)  # encodage
                    ## création des target et features à partir du dataset
                    with col1_target:
                        st.session_state.target = st.selectbox("Target :",
                                                               ["Selectionner une target"] + col_numeric(df_ml),
                                                               )
                    if st.session_state.target != "Selectionner une target":
                        ## KNN
                        st.write("##")
                        y_knn = df_ml[st.session_state.target]  # target
                        X_knn = df_ml.drop(st.session_state.target, axis=1)  # features
                        X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn,
                                                                            test_size=0.4,
                                                                            random_state=4)
                        # Gridsearchcv
                        params = {'n_neighbors': np.arange(1, 20)}
                        grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=4)
                        grid.fit(X_train.values, y_train.values)
                        best_k = grid.best_params_['n_neighbors']
                        best_model_knn = grid.best_estimator_
                        best_model_knn.fit(X_knn.values, y_knn.values)  # on entraine le modèle

                        # Meilleurs hyper params
                        with col_best_score:
                            st.write("##")
                            st.write("##")
                            st.markdown('<p class="section">Sélection des meilleurs hyper-paramètres</p>',
                                        unsafe_allow_html=True)
                            st.write("##")
                            st.success(
                                f'Après un GridSearchCV on prendra **k = {best_k}** voisins')
                            st.write("##")

                        # Évaluation du modèle
                        y_pred_test = best_model_knn.predict(X_test.values)
                        y_pred_train = best_model_knn.predict(X_train.values)
                        if len(y_knn.unique()) > 2:
                            with col_titre_eval_model:
                                st.write("##")
                                st.markdown(
                                    '<p class="section">Évaluation par rapport au train set</p>',
                                    unsafe_allow_html=True)
                                st.write("##")
                                # average='micro' car nos label contiennent plus de 2 classes
                                # Test set
                                precis_test = precision_score(y_test, y_pred_test, average='micro')
                                rappel_test = recall_score(y_test, y_pred_test, average='micro')
                                F1_test = f1_score(y_test, y_pred_test, average='micro')
                                accur_test = accuracy_score(y_test, y_pred_test)
                                # Train set
                                precis_train = precision_score(y_train, y_pred_train, average='micro')
                                rappel_train = recall_score(y_train, y_pred_train, average='micro')
                                F1_train = f1_score(y_train, y_pred_train, average='micro')
                                accur_train = accuracy_score(y_train, y_pred_train)
                                with col1_eval_modele:
                                    st.metric(label="Precision", value=round(precis_test, 3),
                                              delta=round(precis_test - precis_train, 3))
                                with col2_eval_modele:
                                    st.metric(label="Recall", value=round(rappel_test, 3),
                                              delta=round(rappel_test - rappel_train, 3))
                                with col3_eval_modele:
                                    st.metric(label="F1 score", value=round(F1_test, 3),
                                              delta=round(F1_test - F1_train, 3))
                                with col4_eval_modele:
                                    st.metric(label="Accuracy", value=round(accur_test, 3),
                                              delta=round(accur_test - accur_train, 3))

                        else:
                            with col_titre_eval_model:
                                st.write("##")
                                st.markdown(
                                    '<p class="section">Évaluation par rapport au train set</p>',
                                    unsafe_allow_html=True)
                                st.write("##")
                                # label binaire
                                # Test set
                                precis_test = precision_score(y_test, y_pred_test)
                                rappel_test = recall_score(y_test, y_pred_test)
                                F1_test = f1_score(y_test, y_pred_test)
                                accur_test = accuracy_score(y_test, y_pred_test)
                                # Train set
                                precis_train = precision_score(y_train, y_pred_train)
                                rappel_train = recall_score(y_train, y_pred_train)
                                F1_train = f1_score(y_train, y_pred_train)
                                accur_train = accuracy_score(y_train, y_pred_train)
                                with col1_eval_modele:
                                    st.metric(label="Precision", value=round(precis_test, 3),
                                              delta=round(precis_test - precis_train, 3))
                                with col2_eval_modele:
                                    st.metric(label="Recall", value=round(rappel_test, 3),
                                              delta=round(rappel_test - rappel_train, 3))
                                with col3_eval_modele:
                                    st.metric(label="F1 score", value=round(F1_test, 3),
                                              delta=round(F1_test - F1_train, 3))
                                with col4_eval_modele:
                                    st.metric(label="Accuracy", value=round(accur_test, 3),
                                              delta=round(accur_test - accur_train, 3))

                            with col1_roc:
                                st.write("##")
                                st.write("##")
                                st.markdown(
                                    '<p class="section">ROC curve</p>',
                                    unsafe_allow_html=True)
                                fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
                                with col1_AUC_value:
                                    st.write("##")
                                    st.info(f'Area Under the Curve (AUC) = {round(auc(fpr, tpr), 4)}')
                                fig = px.area(
                                    x=fpr, y=tpr,
                                    labels=dict(x='False Positive Rate', y='True Positive Rate'),
                                    width=500, height=500,
                                )
                                fig.add_shape(
                                    type='line', line=dict(dash='dash'),
                                    x0=0, x1=1, y0=0, y1=1
                                )

                                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                                fig.update_xaxes(constrain='domain')
                                fig.update_layout(
                                    font=dict(size=10),
                                    autosize=False,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    width=1050, height=650,
                                    margin=dict(l=40, r=50, b=40, t=40),
                                )
                                st.plotly_chart(fig)

                        # Learning curve
                        with col_learning_curve:
                            st.write("##")
                            st.markdown(
                                '<p class="section">Learning curves</p>',
                                unsafe_allow_html=True)
                            st.write("##")
                            N, train_score, val_score = learning_curve(best_model_knn, X_train, y_train,
                                                                       train_sizes=np.linspace(0.2,
                                                                                               1.0,
                                                                                               10),
                                                                       cv=3, random_state=4)
                            fig = go.Figure()
                            fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train',
                                            marker=dict(color='deepskyblue'))
                            fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation', marker=dict(color='red'))
                            fig.update_layout(
                                showlegend=True,
                                template='simple_white',
                                font=dict(size=10),
                                autosize=False,
                                width=1250, height=650,
                                margin=dict(l=40, r=50, b=40, t=40),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                            )
                            st.plotly_chart(fig)
                            st.caption(
                                "Il est possible que votre dataset soit trop petit pour effectuer la cross-validation dans de bonnes conditions")

                        # Faire une prédiction
                        with col_data_to_predict:
                            st.write("##")
                            st.write("##")
                            st.markdown('---')
                            st.write("##")
                            st.markdown(
                                '<p class="section">Prédiction à l\'aide du modèle</p>',
                                unsafe_allow_html=True)
                            st.write("##")
                            st.write("##")
                            features = []
                            st.markdown('<p class="section">Entrez vos données</p>', unsafe_allow_html=True)
                            st.write("##")
                            for col in X_test.columns.tolist():
                                col = st.text_input(col)
                                features.append(col)
                        if "" not in features:
                            prediction_knn = best_model_knn.predict(np.array(features, dtype=float).reshape(1, -1))
                            with sub_col_prediction_knn:
                                st.write("##")
                                st.success(
                                    f'Prédiction de la target {st.session_state.target} avec les données entrées : **{str(df_origine[st.session_state.target].unique()[int(prediction_knn[0])])}**')
                                st.write("##")

        else:
            with exp2:
                st.write("##")
                st.info('Rendez-vous dans la section Dataset pour importer votre dataset')


    elif st.session_state.choix_page_ml == "K-Means":
        st.markdown('<p class="grand_titre">K-Means</p>', unsafe_allow_html=True)
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        with exp2:
            with st.expander("Principe de l'algorithme des K-means"):
                st.write("""
                Le but est de diviser les points en k clusters.
                * 1ère étape : On place au hasard les k centroïdes
                * 2ème étape : Pour chaque point, on lui associe l'étiquette du centroïde le plus proche
                * 3ème étape : On calcul le centre de gravité des k clusters que nous venons de créer 
                * 4ème étape : On réitère les étapes 2 et 3 jusqu'à temps que les centroïdes ne bougent plus
                
                On peut utiliser diverses techniques pour choisir les premiers centroïdes, et diverses métriques 
                pour évaluer les distances.
                """)
        if 'data' in st.session_state:
            _, col1_features_choice, _ = st.columns((0.1, 1, 0.1))
            with col1_features_choice:
                st.write("##")
                st.markdown('<p class="section">Selection des features pour le modèle</p>', unsafe_allow_html=True)
                st.session_state.choix_col_kmeans = st.multiselect("Choisir au moins deux colonnes",
                                                                   col_numeric(st.session_state.data),
                                                                   )
            if len(st.session_state.choix_col_kmeans) > 1:
                df_ml = st.session_state.data[st.session_state.choix_col_kmeans]
                df_ml = df_ml.dropna(axis=0)
                if len(df_ml) == 0:
                    with col1_features_choice:
                        st.write("##")
                        st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else:
                    with col1_features_choice:
                        X = df_ml[st.session_state.choix_col_kmeans]  # features
                        try:
                            ## PCA
                            model = PCA(n_components=2)
                            model.fit(X)
                            x_pca = model.transform(X)

                            df = pd.concat([pd.Series(x_pca[:, 0]).reset_index(drop=True),
                                            pd.Series(x_pca[:, 1]).reset_index(drop=True)], axis=1)
                            df.columns = ["x", "y"]

                            ## K-Means
                            st.write("##")
                            st.markdown('<p class="section">Résultats</p>', unsafe_allow_html=True)
                            st.session_state.cluster = st.slider('Nombre de clusters', min_value=2,
                                                                 max_value=int(len(X) * 0.2),
                                                                 )
                            X_pca_kmeans = df

                            modele = KMeans(n_clusters=st.session_state.cluster)
                            modele.fit(X_pca_kmeans)
                            y_kmeans = modele.predict(X_pca_kmeans)
                            df["class"] = pd.Series(y_kmeans)

                            fig = px.scatter(df, x=X_pca_kmeans['x'], y=X_pca_kmeans['y'], color="class",
                                             color_discrete_sequence=px.colors.qualitative.G10)
                            fig.update_layout(
                                showlegend=True,
                                template='simple_white',
                                font=dict(size=10),
                                autosize=False,
                                width=1000, height=650,
                                margin=dict(l=40, r=50, b=40, t=40),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                title="K-Means avec " + str(st.session_state.cluster) + " Cluster",
                            )
                            fig.update(layout_coloraxis_showscale=False)
                            centers = modele.cluster_centers_
                            fig.add_scatter(x=centers[:, 0], y=centers[:, 1], mode='markers',
                                            marker=dict(color='black', size=15), opacity=0.4, name='Centroïdes')
                            st.write("##")
                            st.markdown(
                                '<p class="section">Visualisation grâce à une réduction de dimensions (PCA)</p>',
                                unsafe_allow_html=True)
                            st.write("##")
                            st.plotly_chart(fig)
                        except:
                            with col1:
                                st.write("##")
                                st.error("Erreur de chargement")
        else:
            with exp2:
                st.write("##")
                st.info('Rendez-vous dans la section Dataset pour importer votre dataset')


    elif st.session_state.choix_page_ml == "Support Vector Machine":
        st.markdown('<p class="grand_titre">SVM : Support Vector Machine</p>', unsafe_allow_html=True)
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        with exp2:
            with st.expander("Principe des Support Vector Machine"):
                st.write("""
                Le but est de séparer les classes à l'aide d'une droite/courbe qui va maximiser sa distance avec les points les plus proches (les support vectors).
                
                On peut pour cela utiliser différents noyaux qui peuvent être linéaire ou polynomial.
                """)
        if 'data' in st.session_state:
            st.write("##")
            st.markdown('<p class="section">Selection des features et de la target</p>', unsafe_allow_html=True)
            _, col1_km, _ = st.columns((0.1, 1, 0.1))
            with col1_km:
                st.session_state.choix_col_SVM = st.multiselect("Choisir deux colonnes",
                                                                col_numeric(st.session_state.data),
                                                                )
                st.session_state.choix_target_SVM = st.selectbox("Choisir la target",
                                                                 st.session_state.data.columns.tolist(),
                                                                 )

            if len(st.session_state.choix_col_SVM) == 2:
                target = st.session_state.choix_target_SVM
                features = st.session_state.choix_col_SVM

                # dataset avec features + target
                df = st.session_state.data[[target] + features]
                df.dropna(axis=0)

                if len(df) == 0:
                    with col1_km:
                        st.write("##")
                        st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else:
                    if st.session_state.choix_target_SVM in st.session_state.choix_col_SVM:
                        with col1_km:
                            st.warning("La target ne doit pas appartenir aux features")
                    else:
                        if len(df[target].unique().tolist()) > 1:
                            with col1_km:
                                st.session_state.classes_SVM = st.multiselect("Choisir deux classes",
                                                                              df[
                                                                                  st.session_state.choix_target_SVM].unique().tolist(), )
                                if len(st.session_state.classes_SVM) > 1:
                                    df = df.loc[
                                        (df[target] == st.session_state.classes_SVM[0]) | (
                                                df[target] == st.session_state.classes_SVM[1])]
                                    y = df[target]
                                    X = df[features]
                                    st.session_state.choix_kernel = st.selectbox("Choisir le type de noyau",
                                                                                 ['Linéaire'],
                                                                                 )

                                    if st.session_state.choix_kernel == 'Linéaire':
                                        fig = px.scatter(df, x=features[0], y=features[1], color=target,
                                                         color_continuous_scale=px.colors.diverging.Picnic)
                                        fig.update(layout_coloraxis_showscale=False)

                                        from sklearn.svm import SVC  # "Support vector classifier"

                                        model = SVC(kernel='linear', C=1E10)
                                        model.fit(X, y)  # to do ajouter un gridsearchcv

                                        # Support Vectors
                                        fig.add_scatter(x=model.support_vectors_[:, 0],
                                                        y=model.support_vectors_[:, 1],
                                                        mode='markers',
                                                        name="Support vectors",
                                                        marker=dict(size=12,
                                                                    line=dict(width=1,
                                                                              color='DarkSlateGrey'
                                                                              ),
                                                                    color='rgba(0,0,0,0)'),
                                                        )

                                        # hyperplan
                                        w = model.coef_[0]
                                        a = -w[0] / w[1]
                                        xx = np.linspace(df[features[0]].min(), df[features[0]].max())
                                        yy = a * xx - (model.intercept_[0]) / w[1]
                                        fig.add_scatter(x=xx, y=yy, line=dict(color='black', width=2),
                                                        name='Hyperplan')

                                        # Hyperplans up et down
                                        b = model.support_vectors_[0]
                                        yy_down = a * xx + (b[1] - a * b[0])
                                        fig.add_scatter(x=xx, y=yy_down,
                                                        line=dict(color='black', width=1, dash='dot'),
                                                        name='Marges')
                                        b = model.support_vectors_[-1]
                                        yy_up = a * xx + (b[1] - a * b[0])
                                        fig.add_scatter(x=xx, y=yy_up,
                                                        line=dict(color='black', width=1, dash='dot'),
                                                        showlegend=False)
                                        fig.update_layout(
                                            showlegend=True,
                                            template='simple_white',
                                            font=dict(size=10),
                                            autosize=False,
                                            width=1000, height=650,
                                            margin=dict(l=40, r=50, b=40, t=40),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                        )
                                        with col1_km:
                                            st.write("##")
                                            st.plotly_chart(fig)

                                elif len(st.session_state.classes_SVM) > 2:
                                    with col1_km:
                                        st.warning("Saisie invalide - trop de colonne selectionnées")

                        else:
                            with col1_km:
                                st.warning("Le dataset ne contient qu'une classe")
            elif len(st.session_state.choix_col_SVM) > 2:
                with col1_km:
                    st.warning("Saisie invalide - trop de colonne selectionnées")


        else:
            with exp2:
                st.write("##")
                st.info('Rendez-vous dans la section Dataset pour importer votre dataset')


    elif st.session_state.choix_page_ml == "PCA":
        st.markdown('<p class="grand_titre">PCA : Analyse en composantes principales</p>', unsafe_allow_html=True)
        st.write("##")
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        with exp2:
            with st.expander("Principe du PCA"):
                st.write("""
                Le PCA s'applique sur des variables quantitatives de même importance.
                * 1ère étape : on détermine la matrice de covariance entre chaque variable
                * 2ème étape : Par normalisation de cette dernière, on obtient la matrice de corrélation
                * 3ème étape : Par diagonalisation on trouve les elements propres
                * 4ème étape : On détermine les composantes principales
                * 5ème étape : On calcul la matrice des saturations
                * 6ème étape : Représentation graphique
                """)
        if 'data' in st.session_state:
            _, col1_pca, _ = st.columns((0.1, 1, 0.1))
            _, sub_col1, _ = st.columns((0.4, 0.5, 0.4))
            _, col2_pca, _ = st.columns((0.1, 1, 0.1))
            with col1_pca:
                st.write("##")
                st.markdown('<p class="section">Selection des colonnes pour le modèle PCA (target+features)</p>',
                            unsafe_allow_html=True)
                st.session_state.choix_col_PCA = st.multiselect("Choisir au moins deux colonnes",
                                                                st.session_state.data.columns.tolist(),
                                                                )
            if len(st.session_state.choix_col_PCA) > 1:
                df_ml = st.session_state.data[st.session_state.choix_col_PCA]
                df_ml = df_ml.dropna(axis=0)
                st.session_state.df_ml_origine = df_ml.copy()
                if len(df_ml) == 0:
                    with col1_pca:
                        st.write("##")
                        st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else:
                    with col1_pca:
                        # encodage !
                        st.session_state.col_to_encodage_PCA = st.multiselect("Selectionner les colonnes à encoder",
                                                                              st.session_state.choix_col_PCA)
                    with sub_col1:
                        with st.expander('Encodage'):
                            for col in st.session_state.col_to_encodage_PCA:
                                st.write(
                                    "encodage colonne " + col + " : " + str(df_ml[col].unique().tolist()) + "->" + str(
                                        np.arange(len(df_ml[col].unique()))))
                                df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())),
                                                   inplace=True)  # encodage
                        ## on choisit notre modèle
                        model = PCA(n_components=2)
                    with col2_pca:
                        ## création des target et features à partir du dataset
                        st.write("##")
                        st.write("##")
                        st.session_state.target_PCA = st.selectbox("Target :",
                                                                   ["Selectionner une target"] + col_numeric(df_ml),
                                                                   )
                    if st.session_state.target_PCA != "Selectionner une target":
                        y = df_ml[st.session_state.target_PCA]  # target
                        X = df_ml.drop(st.session_state.target_PCA, axis=1)  # features

                        try:
                            with col2_pca:
                                model.fit(X)
                                x_pca = model.transform(X)
                                st.write("##")
                                st.markdown('<p class="section">Résultats</p>', unsafe_allow_html=True)
                                # résultats points
                                st.session_state.df = pd.concat([pd.Series(x_pca[:, 0]).reset_index(drop=True),
                                                                 pd.Series(x_pca[:, 1]).reset_index(drop=True),
                                                                 pd.Series(st.session_state.df_ml_origine[
                                                                               st.session_state.target_PCA]).reset_index(
                                                                     drop=True)], axis=1)
                                st.session_state.df.columns = ["x", "y", str(st.session_state.target_PCA)]
                                fig = px.scatter(st.session_state.df, x="x", y="y",
                                                 color=str(st.session_state.target_PCA),
                                                 labels={'color': '{}'.format(str(st.session_state.target_PCA))},
                                                 color_discrete_sequence=px.colors.qualitative.Plotly)
                                fig.update_layout(
                                    showlegend=True,
                                    template='simple_white',
                                    font=dict(size=10),
                                    autosize=False,
                                    width=1050, height=650,
                                    margin=dict(l=40, r=50, b=40, t=40),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                )
                                fig.update(layout_coloraxis_showscale=False)
                                st.plotly_chart(fig)
                        except:
                            st.write("##")
                            st.error("Erreur de chargement!")
        else:
            with exp2:
                st.write("##")
            st.info('Rendez-vous dans la section Dataset pour importer votre dataset')


    elif st.session_state.choix_page_ml == "UMAP":
        st.markdown('<p class="grand_titre">UMAP : Uniform Manifold Approximation and Projection</p>',
                    unsafe_allow_html=True)
        exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
        if 'data' in st.session_state:
            _, col1_umap, _ = st.columns((0.1, 1, 0.1))
            _, sub_col1, _ = st.columns((0.4, 0.5, 0.4))
            _, col2_umap, _ = st.columns((0.1, 1, 0.1))
            with col1_umap:
                st.write("##")
                st.markdown('<p class="section">Selection des colonnes pour le modèle UMAP (target+features)</p>',
                            unsafe_allow_html=True)
                st.session_state.choix_col_UMAP = st.multiselect("Choisir au moins deux colonnes",
                                                                 st.session_state.data.columns.tolist(),
                                                                 )
            if len(st.session_state.choix_col_UMAP) > 1:
                df_ml = st.session_state.data[st.session_state.choix_col_UMAP]
                df_ml = df_ml.dropna(axis=0)
                st.session_state.df_ml_origine = df_ml.copy()
                if len(df_ml) == 0:
                    with col1_umap:
                        st.write("##")
                        st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else:
                    with col1_umap:
                        # encodage !
                        st.session_state.col_to_encodage_UMAP = st.multiselect("Selectionner les colonnes à encoder",
                                                                               st.session_state.choix_col_UMAP,
                                                                               )
                    with sub_col1:
                        with st.expander('Encodage'):
                            for col in st.session_state.col_to_encodage_UMAP:
                                st.write(
                                    "encodage colonne " + col + " : " + str(df_ml[col].unique().tolist()) + "->" + str(
                                        np.arange(len(df_ml[col].unique()))))
                                df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())),
                                                   inplace=True)  # encodage

                    with col2_umap:
                        ## on choisit notre modèle
                        model = UMAP(random_state=0)
                        ## création des target et features à partir du dataset
                        st.write("##")
                        st.session_state.target_UMAP = st.selectbox("Target :",
                                                                    ["Selectionner une target"] + col_numeric(df_ml),
                                                                    )
                    if st.session_state.target_UMAP != "Selectionner une target":
                        y = df_ml[st.session_state.target_UMAP]  # target
                        X = df_ml.drop(st.session_state.target_UMAP, axis=1)  # features
                        with col2_umap:
                            try:
                                model.fit(X)
                                x_umap = model.transform(X)
                                st.write("##")
                                st.write("##")
                                st.markdown('<p class="section">Résultats</p>', unsafe_allow_html=True)
                                # résultats points
                                st.session_state.df = pd.concat([pd.Series(x_umap[:, 0]), pd.Series(x_umap[:, 1]),
                                                                 pd.Series(st.session_state.df_ml_origine[
                                                                               st.session_state.target_UMAP])], axis=1)
                                st.session_state.df.columns = ["x", "y", str(st.session_state.target_UMAP)]
                                fig = px.scatter(st.session_state.df, x="x", y="y",
                                                 color=str(st.session_state.target_UMAP),
                                                 labels={'color': '{}'.format(str(st.session_state.target_UMAP))},
                                                 color_discrete_sequence=px.colors.qualitative.Plotly)
                                fig.update_layout(
                                    showlegend=True,
                                    template='simple_white',
                                    font=dict(size=10),
                                    autosize=False,
                                    width=1250, height=650,
                                    margin=dict(l=40, r=50, b=40, t=40),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                )
                                fig.update(layout_coloraxis_showscale=False)
                                st.plotly_chart(fig)
                            except:
                                st.write("##")
                                st.error("Erreur de chargement!")
        else:
            with exp2:
                st.write("##")
                st.info('Rendez-vous dans la section Dataset pour importer votre dataset')
############# ML section #############
