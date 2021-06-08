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

uploaded_file = st.sidebar.file_uploader("Chargez votre dataset",type=['csv','xls'])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.sidebar.success('Fichier chargé')
    if 'csv' in file_details['FileType']:
        st.sidebar.write('Type : csv')
        data = pd.read_csv(uploaded_file)
    else :
        st.sidebar.write('Type : excel')
        data = pd.read_excel(uploaded_file)

    st.write("##")
    st.write("Aperçu du dataset : ")
    st.write(data.head(50))

    st.write("##")
    st.write(' ● size:', data.shape)
    st.write(' ● data type:', data.dtypes.value_counts())
    st.write(' ● missing values:', round(sum(pd.DataFrame(data).isnull().sum(axis=1).tolist())*100/(data.shape[0]*data.shape[1]),2),' % (', sum(pd.DataFrame(data).isnull().sum(axis=1).tolist()),')' )
    st.write(' ● number of values:', data.shape[0]*data.shape[1])

