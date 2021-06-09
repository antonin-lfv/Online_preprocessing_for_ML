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
st.set_page_config(layout="wide")

###### functions ########
def max_std(dataset):
    l = []
    for nom in dataset.columns :
        if type(dataset[nom][0]) != object and type(dataset[nom][0]) != str :
            l.append([dataset[nom].std(),nom])
    return(max(l))

####### html/css config ########
st.markdown("""
<style>
.first_titre {
    font-size:50px !important;
    font-weight: bold;
    box-sizing: border-box;
    text-align: center;
    width: 100%;
    border: solid #CD2727 5px;
    padding: 5px;
}
.intro{
    text-align: justify;
}
.grand_titre {
    font-size:30px !important;
    font-weight: bold;
    text-decoration: underline;
    text-decoration-color: #2782CD;
    text-decoration-thickness: 5px;
}
.section{
    font-size:20px !important;
    text-decoration: underline;
}
.caract{
    font-size:10px !important;
}
</style>
""", unsafe_allow_html=True)

####### Streamlit home ######
st.markdown('<p class="first_titre">Preprocessing automatique</p>', unsafe_allow_html=True)
st.write("##")
st.markdown('<p class="intro">Bienvenue sur le site de Preprocessing en ligne ! Déposez vos datasets csv et excel et commencez votre analyse dès maintenant ! Cherchez les variables les plus intéressantes pour développer votre modèle, ou simplement pour visualiser vos données.</p>',unsafe_allow_html=True)


###### App #######
uploaded_file = st.sidebar.file_uploader("Chargez votre dataset",type=['csv','xls'])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    try :
        if 'csv' in file_details['FileName']:
            data = pd.read_csv(uploaded_file)
            slider_col = st.sidebar.selectbox(
                'Choisissez une colonne à étudier',
                ['Choisir'] + data.columns.to_list(),
            )
        else:
            data = pd.read_excel(uploaded_file)
            slider_col = st.sidebar.selectbox(
                'Choisissez une colonne à étudier',
                ['Choisir'] + data.columns.to_list(),
            )

        st.sidebar.success('Fichier chargé')
        ### section du dataset ###
        st.write("##")
        st.markdown('<p class="grand_titre">Analyse du dataset</p>', unsafe_allow_html=True)
        st.write("##")
        st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
        st.write(data.head(50))
        st.write("##")

        st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
        st.write(' - Taille:', data.shape)
        st.write(' - Nombre de valeurs:', data.shape[0] * data.shape[1])
        st.write(' - Type des colonnes:', data.dtypes.value_counts())
        st.write(' - Pourcentage de valeurs manquantes:', round(
            sum(pd.DataFrame(data).isnull().sum(axis=1).tolist()) * 100 / (data.shape[0] * data.shape[1]), 2),
                 ' % (', sum(pd.DataFrame(data).isnull().sum(axis=1).tolist()), ' valeurs manquantes)')

        ### Section de la colonne ###
        if slider_col != 'Choisir':
            st.write('##')
            st.markdown('<p class="grand_titre">Analyse de la colonne '+slider_col+'</p>', unsafe_allow_html=True)
            ### Données ###
            data_col = data[slider_col].copy()
            n_data = data[slider_col].to_numpy()

            st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
            st.write(' ● aperçu des données:')
            st.write(data_col.head(20))
            st.write(' ● type de la colonne :', type(data_col))
            if n_data.dtype == float:
                moyenne = data_col.mean()
                variance = data_col.std()
                max = data_col.max()
                min = data_col.min()
                st.write(' ● Moyenne :', moyenne)

                st.write(' ● Variance :', variance)

                st.write(' ● Maximum :', max)

                st.write(' ● Minimum :', min)

            st.write(' ● Valeurs les plus présentes:', (Counter(n_data).most_common()[0])[0], '->',
                     (Counter(n_data).most_common()[0])[1], 'fois', ', ', (Counter(n_data).most_common()[1])[0], '->',
                     (Counter(n_data).most_common()[1])[1], 'fois')

            st.write(' ● Nombre de valeurs manquantes:', sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))

            st.write(' ● Longueur:', n_data.shape[0])

            st.write(' ● Nombre de valeurs différentes non NaN:',
                     abs(len(Counter(n_data)) - sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
            st.write("##")
            ### Fin section données
        ### Fin section colonne
    except:
        st.sidebar.error('Erreur de chargement')
