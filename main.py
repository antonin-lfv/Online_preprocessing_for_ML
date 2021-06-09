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

###### functions ########
def max_std(dataset):
    l = []
    for nom in dataset.columns :
        if type(dataset[nom][0]) != object and type(dataset[nom][0]) != str :
            l.append([dataset[nom].std(),nom])
    return(max(l))


####### Streamlit Config ######
st.set_page_config(layout="wide", )
st.title('Preprocessing automatique')


###### App #######
uploaded_file = st.sidebar.file_uploader("Chargez votre dataset",type=['csv','xls'])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.sidebar.success('Fichier chargé')
    if 'csv' in file_details['FileName']:
        try :
            data = pd.read_csv(uploaded_file)
            slider_col = st.sidebar.selectbox(
                'Choisissez une colonne à étudier',
                ['Choisir']+data.columns.to_list(),
            )
            ### section du dataset ###
            st.write("##")
            st.write("Aperçu du dataset : ")
            st.write(data.head(50))

            st.write("##")
            st.write(' ● size:', data.shape)
            st.write(' ● data type:', data.dtypes.value_counts())
            st.write(' ● missing values:', round(
                sum(pd.DataFrame(data).isnull().sum(axis=1).tolist()) * 100 / (data.shape[0] * data.shape[1]), 2),
                     ' % (', sum(pd.DataFrame(data).isnull().sum(axis=1).tolist()), ' valeurs manquantes)')
            st.write(' ● number of values:', data.shape[0] * data.shape[1])
            ### Section de la colonne ###
            if slider_col!='Choisir':
                st.title('Étude la colonne {}'.format(slider_col))
                ### Données ###
                n_data = (data[slider_col].to_numpy())

                st.write(' ● données:')
                st.write(data.head(5))
                st.write("##")
                st.write(' ● data type :', type(data))
                st.write("##")
                if n_data.dtype == float:
                    moyenne = data.mean()
                    variance = data.std()
                    max = data.max()
                    min = data.min()
                    st.write(' ● average :', moyenne)
                    st.write("##")

                    st.write(' ● variance :', variance)
                    st.write("##")

                    st.write(' ● maximum :', max)
                    st.write("##")

                    st.write(' ● minimum :', min)
                    st.write("##")

                st.write(' ● most present value:', (Counter(n_data).most_common()[0])[0], ':',
                         (Counter(n_data).most_common()[0])[1], 'fois', ',', (Counter(n_data).most_common()[1])[0], ':',
                         (Counter(n_data).most_common()[1])[1], 'fois')
                st.write("##")

                st.write(' ● missing values:', sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))
                st.write("##")

                st.write(' ● length:', n_data.shape[0])
                st.write("##")

                st.write(' ● number of different values not NaN:',
                         abs(len(Counter(n_data)) - sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
                st.write("##")
                ### Fin section données
            ### Fin section colonne

        except :
            st.sidebar.error('Erreur de chargement')
    else :
        try :
            data = pd.read_excel(uploaded_file)
            slider_col = st.sidebar.selectbox(
                'Choisissez une colonne à étudier',
                data.columns.to_list(),
            )
            st.write("##")
            st.write("Aperçu du dataset : ")
            st.write(data.head(50))

            st.write("##")
            st.write(' ● size:', data.shape)
            st.write(' ● data type:', data.dtypes.value_counts())
            st.write(' ● missing values:', round(
                sum(pd.DataFrame(data).isnull().sum(axis=1).tolist()) * 100 / (data.shape[0] * data.shape[1]), 2),
                     ' % (', sum(pd.DataFrame(data).isnull().sum(axis=1).tolist()), ' valeurs manquantes)')
            st.write(' ● number of values:', data.shape[0] * data.shape[1])
        except :
            st.sidebar.error('Erreur de chargement')


