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
import plotly.graph_objects as go
import webbrowser

st.set_page_config(layout="wide")


###### functions ########
def max_std(dataset):
    l = []
    for nom in dataset.columns:
        if type(dataset[nom][0]) != object and type(dataset[nom][0]) != str:
            l.append([dataset[nom].std(), nom])
    return (max(l))


def col_numeric(df):
    return df.select_dtypes(include=np.number).columns.tolist()


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
st.markdown(
    '<p class="intro">Bienvenue sur le site de Preprocessing en ligne ! Déposez vos datasets csv et excel et commencez votre analyse dès maintenant ! Cherchez les variables les plus intéressantes pour développer votre modèle, ou simplement pour visualiser vos données. '+
    'Si vous aimez ce site n\'hésitez pas à mettre une étoile sur le repo GitHub.</p>',
    unsafe_allow_html=True)
url = 'https://github.com/antonin-lfv/Online_preprocessing_for_ML'
if st.button('Github project'):
    webbrowser.open_new_tab(url)

###### App #######
uploaded_file = st.sidebar.file_uploader("Chargez votre dataset", type=['csv', 'xls'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    try:
        if 'csv' in file_details['FileName']:
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.sidebar.success('Fichier chargé avec succès !')
        st.sidebar.write('##')
        st.sidebar.write('Affichage')
        analyse_data = st.sidebar.checkbox('Analyse du dataset', value=True)
        analyse_col = st.sidebar.checkbox('Analyse d\'une colonne')
        analyses_graph = st.sidebar.checkbox('Analyses graphiques')

        if analyse_data:
            ##########################
            ### section du dataset ###
            ##########################
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
            ### Fin section du dataset ###

        if analyse_col:
            #############################
            ### Section de la colonne ###
            #############################
            st.write('##')
            st.markdown('<p class="grand_titre">Analyse d\'une colonne</p>', unsafe_allow_html=True)
            slider_col = st.selectbox(
                'Choisissez une colonne à étudier',
                ['Selectionner une colonne'] + data.columns.to_list(),
            )
            if slider_col != 'Selectionner une colonne':
                ### Données ###
                data_col = data[slider_col].copy()
                n_data = data[slider_col].to_numpy()

                st.write('##')
                st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                st.write(data_col.head(20))
                st.write("##")

                st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                st.write(' ● type de la colonne :', type(data_col))
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

                st.write(' ● Nombre de valeurs manquantes:', sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))

                st.write(' ● Longueur:', n_data.shape[0])

                st.write(' ● Nombre de valeurs différentes non NaN:',
                         abs(len(Counter(n_data)) - sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
                st.write("##")
                ### Fin section données ###
            ### Fin section colonne ###

        if analyses_graph:
            ##########################
            ### Section Graphiques ###
            ##########################
            st.write("##")
            st.markdown('<p class="grand_titre">Analyses graphiques</p>', unsafe_allow_html=True)
            abscisse_plot = st.selectbox('Données en abscisses', ['Selectionner une colonne'] + col_numeric(data))
            ordonnee_plot = st.selectbox('Données en ordonnées', ['Selectionner une colonne'] + col_numeric(data))
            couleur_plot = st.selectbox('Couleur', ['Selectionner une colonne'] + data.columns.tolist())
            type_plot = st.radio("Type de plot", ('Points', 'Courbe', 'Latitude/Longitude'))
            type_plot_dict = {
                'Courbe': 'lines',
                'Points': 'markers',
                'Latitude/Longitude': 'map'
            }
            if abscisse_plot != 'Selectionner une colonne' and ordonnee_plot != 'Selectionner une colonne':
                df_sans_NaN = pd.concat([data[abscisse_plot], data[ordonnee_plot]], axis=1).dropna()
                if type_plot == 'Latitude/Longitude':
                    fig = go.Figure()
                    if couleur_plot!='Selectionner une colonne':
                        fig.add_scattermapbox(
                            mode="markers",
                            lon=df_sans_NaN[ordonnee_plot],
                            lat=df_sans_NaN[abscisse_plot],
                            marker={'size': 10,
                                    'color': data[couleur_plot],
                                    })
                    else :
                        fig.add_scattermapbox(
                            mode="markers",
                            lon=df_sans_NaN[ordonnee_plot],
                            lat=df_sans_NaN[abscisse_plot],
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

                else:
                    fig = go.Figure()
                    if couleur_plot != 'Selectionner une colonne' :
                        fig.add_scatter(x=df_sans_NaN[abscisse_plot], y=df_sans_NaN[ordonnee_plot],
                                    mode=type_plot_dict[type_plot], name='', marker=dict(color=data[couleur_plot]))
                    else :
                        fig.add_scatter(x=df_sans_NaN[abscisse_plot], y=df_sans_NaN[ordonnee_plot],
                                    mode=type_plot_dict[type_plot], name='')
                    fig.update_xaxes(title_text=abscisse_plot)
                    fig.update_yaxes(title_text=ordonnee_plot)
                    fig.update_layout(
                        template='simple_white',
                        showlegend=False,
                        font=dict(size=10),
                        autosize=False,
                        width=900, height=450,
                        margin=dict(l=40, r=50, b=40, t=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig)
            ### Fin section graphiques ###
    except:
        st.sidebar.error('Erreur de chargement')