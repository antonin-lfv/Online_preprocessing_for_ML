# Importations
import streamlit as st
import plotly.express as px
import numpy as np
from collections import Counter
import pandas as pd
import time
import os
import plotly.graph_objects as go
import webbrowser
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# streamlit run main.py

####### html/css config ########
st.set_page_config(layout="wide")
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


###### functions ########
def max_std(dataset):
    l = []
    for nom in dataset.columns:
        if type(dataset[nom][0]) != object and type(dataset[nom][0]) != str:
            l.append([dataset[nom].std(), nom])
    return (max(l))

def col_numeric(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def clean_data(x):
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', '').replace('€', '').replace('£', ''))
    return(x)

###### Session data ######


####### Streamlit home + upload file ######
st.markdown('<p class="first_titre">Preprocessing automatique</p>', unsafe_allow_html=True)
st.write("##")

uploaded_file = st.sidebar.file_uploader("Chargez votre dataset", type=['csv', 'xls'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    separateur = st.sidebar.text_input("Séparateur (optionnel): ")
    try:
        if 'csv' in file_details['FileName']:
            if separateur != "":
                data = pd.read_csv(uploaded_file, sep=separateur)
            else :
                data = pd.read_csv(uploaded_file)
        else:
            if separateur != "":
                data = pd.read_excel(uploaded_file, sep=separateur)
            else :
                data = pd.read_excel(uploaded_file)

        st.sidebar.success('Fichier chargé avec succès !')
    except:
        st.sidebar.error('Erreur de chargement')
else :
    pass


def main():
    PAGES = {
        "Accueil": page1,
        "Analyse du dataset": page2,
        "Analyse d'une colonne": page3,
        "Graphiques et Regressions": page4,
        "Matrice de corrélation": page5,
        "Machine Learning - KNN": page6,
    }
    st.sidebar.write("##")
    st.sidebar.title('Menu')
    st.sidebar.subheader('Data visualisation and ML')
    page=st.sidebar.radio("", list(PAGES.keys()))
    PAGES[page]()



###############
### Accueil ###
###############
def page1():
    st.write("##")
    st.markdown(
        '<p class="intro">Bienvenue sur le site de Preprocessing en ligne ! Déposez vos datasets csv et excel et commencez votre analyse dès maintenant ! Cherchez les variables les plus intéressantes pour développer votre modèle, ou simplement pour visualiser vos données. ' +
        'Si vous aimez ce site n\'hésitez pas à mettre une étoile sur le repo GitHub.</p>',
        unsafe_allow_html=True)
    st.write("Github Project : [here](https://github.com/antonin-lfv/Online_preprocessing_for_ML)")
### Fin accueil ###









##########################
### section du dataset ###
##########################
def page2():
    if uploaded_file is not None:
        st.write("##")
        st.markdown('<p class="grand_titre">Analyse du dataset</p>', unsafe_allow_html=True)
        st.write("##")
        col1, b, col2 = st.beta_columns((2.7, 0.3, 1))
        with col1 :
            st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
            st.write(data.head(50))
            st.write("##")

        with col2 :
            st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
            st.write(' - Taille:', data.shape)
            st.write(' - Nombre de valeurs:', data.shape[0] * data.shape[1])
            st.write(' - Type des colonnes:', data.dtypes.value_counts())
            st.write(' - Pourcentage de valeurs manquantes:', round(
                sum(pd.DataFrame(data).isnull().sum(axis=1).tolist()) * 100 / (data.shape[0] * data.shape[1]), 2),
                     ' % (', sum(pd.DataFrame(data).isnull().sum(axis=1).tolist()), ')')
    else :
        st.warning('Veuillez charger un dataset !')
### Fin section du dataset ###










#############################
### Section de la colonne ###
#############################
def page3():
    if uploaded_file is not None:
        st.write('##')
        st.markdown('<p class="grand_titre">Analyse d\'une colonne</p>', unsafe_allow_html=True)
        col1, b,col2 = st.beta_columns((1,0.3,2))
        with col1 :
            slider_col = st.selectbox(
                'Choisissez une colonne à étudier',
                ['Selectionner une colonne'] + data.columns.to_list(),
            )
        if slider_col != 'Selectionner une colonne' :
            slider_col = slider_col
            ### Données ###
            data_col = data[slider_col].copy()
            n_data = data[slider_col].to_numpy()

            st.write('##')
            col1, b, col2 = st.beta_columns((1,1,2))
            with col1 :
                st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                st.write(data_col.head(20))

            with col2 :
                st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
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

                st.write(' ● Nombre de valeurs manquantes:', sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))

                st.write(' ● Longueur:', n_data.shape[0])

                st.write(' ● Nombre de valeurs différentes non NaN:',
                         abs(len(Counter(n_data)) - sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
                ### Fin section données ###
    else :
        st.warning('Veuillez charger un dataset !')
### Fin section colonne ###











##########################
### Section Graphiques ###
##########################
def page4():
    if uploaded_file is not None:
        st.write("##")
        st.markdown('<p class="grand_titre">Graphiques et regressions</p>', unsafe_allow_html=True)
        st.write("##")
        st.write("Si des colonnes de votre dataset n'apparaissent pas et qu'elles contiennent des dates, des symboles de monnaies ou des virgules qui empêchent le typage float alors selectionnez les ici : ")
        col1_1, b_1, col2_1, c_1, col3_1 = st.beta_columns((1,0.2,1,0.2,1)) # pour time series
        col1, b, col2, c, col3, d, col4 = st.beta_columns((7)) # pour les autres select
        col_num = col_numeric(data)
        st.write("##")
        with col1_1 :
            col_to_time = st.multiselect('Conversion Time Series', ['Selectionner une/des colonne/s'] + data.columns.tolist())
        with col2_1 :
            col_to_float_money = st.multiselect('Conversion Monnaies', ['Selectionner une/des colonne/s'] + data.columns.tolist())
        with col3_1 :
            col_to_float_coma = st.multiselect('Conversion string avec virgules vers float', ['Selectionner une/des colonne/s'] + data.columns.tolist())
        if 'Selectionner une/des colonne/s' not in col_to_time :
            with col1_1:
                for col in col_to_time :
                    try :
                        data[col]=pd.to_datetime(data[col])
                        col_num+=[col]
                        st.success("Transformation effectuée avec succès !")
                    except :
                        st.error("Transformation impossible")
        if 'Selectionner une/des colonne/s' not in col_to_float_money :
            with col2_1:
                for col in col_to_float_money :
                    try :
                        data[col] = data[col].apply(clean_data).astype('float')
                        col_num += [col]
                        st.success("Transformation effectuée avec succès !")
                    except :
                        st.error("Transformation impossible")
        if 'Selectionner une/des colonne/s' not in col_to_float_coma :
            with col3_1:
                for col in col_to_float_coma :
                    try :
                        data[col] = data[col].apply(lambda x: x.replace(',', '.')).astype('float')
                        col_num += [col]
                        st.success("Transformation effectuée avec succès !")
                    except :
                        st.error("Transformation impossible")
        with col1 :
            st.write("##")
            abscisse_plot = st.selectbox('Données en abscisses', ['Selectionner une colonne'] + col_num)
            ordonnee_plot = st.selectbox('Données en ordonnées', ['Selectionner une colonne'] + col_num)
            # couleur_plot = st.selectbox('Couleur', ['Selectionner une colonne'] + data.columns.tolist())
        with col2 :
            st.write("##")
            type_plot = st.radio("Type de plot", ('Points', 'Courbe', 'Latitude/Longitude', 'Histogramme'))
            type_plot_dict = {
                'Courbe': 'lines',
                'Points': 'markers',
                'Latitude/Longitude': 'map',
            }
        st.write('##')
        if abscisse_plot != 'Selectionner une colonne' and ordonnee_plot != 'Selectionner une colonne':
            if type_plot == 'Latitude/Longitude':
                fig = go.Figure()
                df_sans_NaN = pd.concat([data[abscisse_plot], data[ordonnee_plot]], axis=1).dropna()
                if len(df_sans_NaN)==0 :
                    st.error('Le dataset après dropna() est vide !')
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
            elif type_plot=='Histogramme':
                fig=go.Figure()
                df_sans_NaN = pd.concat([data[abscisse_plot], data[ordonnee_plot]], axis=1).dropna()
                if len(df_sans_NaN)==0 :
                    st.error('Le dataset après dropna() est vide !')
                else :
                    fig.add_histogram(x=df_sans_NaN[abscisse_plot], y=df_sans_NaN[ordonnee_plot])

            else:
                with col3:
                    st.write("##")
                    st.write("##")
                    maximum = st.checkbox("Maximum")
                    moyenne = st.checkbox("Moyenne")
                    minimum = st.checkbox("Minimum")
                fig = go.Figure()
                df_sans_NaN = pd.concat([data[abscisse_plot], data[ordonnee_plot]], axis=1).dropna()
                if len(df_sans_NaN)==0 :
                    st.error('Le dataset après dropna() est vide !')
                else :
                    fig.add_scatter(x=df_sans_NaN[abscisse_plot], y=df_sans_NaN[ordonnee_plot],mode=type_plot_dict[type_plot], name='', showlegend=False)
                    if abscisse_plot not in col_to_time and ordonnee_plot not in col_to_time :
                        with col4:
                            st.write("##")
                            if type_plot == 'Points' or type_plot == 'Courbe':
                                st.write("##")
                                trendline = st.checkbox("Regression linéaire")
                                polynom_feat = st.checkbox("Regression polynomiale")
                                if polynom_feat:
                                    degres = st.slider('Degres de la regression polynomiale', min_value=2,
                                                       max_value=100, value=4)
                        if trendline :
                            # regression linaire
                            X = df_sans_NaN[abscisse_plot].values.reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, df_sans_NaN[ordonnee_plot])
                            x_range = np.linspace(X.min(), X.max(), len(df_sans_NaN[ordonnee_plot]))
                            y_range = model.predict(x_range.reshape(-1, 1))
                            fig.add_scatter(x=x_range, y=y_range, name='Regression linéaire', mode='lines', marker=dict(color='red'))
                            # #################
                        if polynom_feat :
                            # regression polynomiale
                            X = df_sans_NaN[abscisse_plot].values.reshape(-1, 1)
                            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                            poly = PolynomialFeatures(degres)
                            poly.fit(X)
                            X_poly = poly.transform(X)
                            x_range_poly = poly.transform(x_range)
                            model = LinearRegression(fit_intercept=False)
                            model.fit(X_poly, df_sans_NaN[ordonnee_plot])
                            y_poly = model.predict(x_range_poly)
                            fig.add_scatter(x=x_range.squeeze(), y=y_poly, name='Polynomial Features', marker=dict(color='green'))
                            # #################
                    if moyenne :
                        # Moyenne #
                        fig.add_hline(y=df_sans_NaN[ordonnee_plot].mean(),
                                      line_dash="dot",
                                      annotation_text="moyenne : {}".format(round(df_sans_NaN[ordonnee_plot].mean(), 1)),
                                      annotation_position="bottom left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
                        pass
                    if minimum :
                        # Minimum #
                        fig.add_hline(y=df_sans_NaN[ordonnee_plot].min(),
                                      line_dash="dot",
                                      annotation_text="minimum : {}".format(round(df_sans_NaN[ordonnee_plot].min(), 1)),
                                      annotation_position="bottom left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
                        pass
                    if maximum :
                        # Maximum #
                        fig.add_hline(y=df_sans_NaN[ordonnee_plot].max(),
                                      line_dash="dot",
                                      annotation_text="maximum : {}".format(round(df_sans_NaN[ordonnee_plot].max(), 1)),
                                      annotation_position="top left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
                        pass
            if len(df_sans_NaN) != 0:
                fig.update_xaxes(title_text=abscisse_plot)
                fig.update_yaxes(title_text=ordonnee_plot)
                fig.update_layout(
                    template='simple_white',
                    font=dict(size=10),
                    autosize=False,
                    width=1300, height=650,
                    margin=dict(l=40, r=50, b=40, t=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig)
    else :
        st.warning('Veuillez charger un dataset !')
### Fin section graphiques ###









###########################
### Section Mat de corr ###
###########################
def page5():
    if uploaded_file is not None:
            st.write("##")
            st.markdown('<p class="grand_titre">Matrice de corrélations</p>', unsafe_allow_html=True)
            col1, b, col2 = st.beta_columns((1, 1, 2))
            df_sans_NaN = data.dropna()
            if len(df_sans_NaN)==0:
                with col1:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else :
                with col1:
                    couleur_corr = st.selectbox('Couleur', ['Selectionner une colonne'] + df_sans_NaN.columns.tolist())
                    st.write("##")
                select_columns_corr = st.multiselect("Choisir au moins deux colonnes",[ "Toutes les colonnes"] + col_numeric(df_sans_NaN))
                if len(select_columns_corr)>1 and "Toutes les colonnes" not in select_columns_corr:
                    if couleur_corr!='Selectionner une colonne':
                        fig=px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN[select_columns_corr]), color=couleur_corr, color_continuous_scale='Bluered_r')
                    else :
                        fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN[select_columns_corr]))
                    fig.update_layout(width=900, height=700,margin=dict(l=40, r=50, b=40, t=40), font=dict(size=7))
                    fig.update_layout({"xaxis" + str(i+1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN[select_columns_corr])))})
                    fig.update_layout({"yaxis" + str(i+1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN[select_columns_corr])))})
                    fig.update_traces(marker=dict(size=2))
                    fig.update_traces(diagonal_visible=False)
                    st.plotly_chart(fig)
                elif select_columns_corr==["Toutes les colonnes"]:
                    if couleur_corr!='Selectionner une colonne':
                        fig=px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN), color=couleur_corr)
                    else :
                        fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN))
                    fig.update_layout({"xaxis" + str(i+1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
                    fig.update_layout({"yaxis" + str(i+1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
                    fig.update_traces(marker=dict(size=2))
                    fig.update_layout(width=900, height=700,margin=dict(l=40, r=50, b=40, t=40),font=dict(size=7))
                    fig.update_traces(marker=dict(size=2))
                    fig.update_traces(diagonal_visible=False)
                    st.plotly_chart(fig)
                elif len(select_columns_corr)>1 and "Toutes les colonnes" in select_columns_corr :
                    st.error("Erreur de saisi !")
                else :
                    pass
    else :
        st.warning('Veuillez charger un dataset !')
### Fin section mat de corr ###



###########################
### Section ML ###
###########################
def page6():
    if uploaded_file is not None:
            st.write("##")
            st.markdown('<p class="grand_titre">Machine Learning - KNN</p>', unsafe_allow_html=True)
            st.write("##")
            st.write("Si des colonnes de votre dataset n'apparaissent pas et qu'elles contiennent des dates, des symboles de monnaies ou des virgules qui empêchent le typage float alors selectionnez les ici : ")
            col1_1, b_1, col2_1, c_1, col3_1 = st.beta_columns((1, 0.2, 1, 0.2, 1))
            with col1_1:
                col_to_time = st.multiselect('Conversion Time Series',
                                             ['Selectionner une/des colonne/s'] + data.columns.tolist())
            with col2_1:
                col_to_float_money = st.multiselect('Conversion Monnaies',
                                                    ['Selectionner une/des colonne/s'] + data.columns.tolist())
            with col3_1:
                col_to_float_coma = st.multiselect('Conversion string avec virgules vers float',
                                                   ['Selectionner une/des colonne/s'] + data.columns.tolist())
            col1, b, col2 = st.beta_columns((1,0.2,1))
            with col1 :
                st.write("##")
                st.markdown('<p class="section">Selection des colonnes</p>', unsafe_allow_html=True)
                choix_col = st.multiselect("Choisir au moins deux colonnes",["Toutes les colonnes"] + data.columns.tolist())
            if 'Selectionner une/des colonne/s' not in col_to_time:
                with col1_1:
                    for col in col_to_time:
                        try:
                            data[col] = pd.to_datetime(data[col])
                            st.success("Transformation effectuée avec succès !")
                        except:
                            st.error("Transformation impossible")
            if 'Selectionner une/des colonne/s' not in col_to_float_money:
                with col2_1:
                    for col in col_to_float_money:
                        try:
                            data[col] = data[col].apply(clean_data).astype('float')
                            st.success("Transformation effectuée avec succès !")
                        except:
                            st.error("Transformation impossible")
            if 'Selectionner une/des colonne/s' not in col_to_float_coma:
                with col3_1:
                    for col in col_to_float_coma:
                        try:
                            data[col] = data[col].apply(lambda x: x.replace(',', '.')).astype('float')
                            st.success("Transformation effectuée avec succès !")
                        except:
                            st.error("Transformation impossible")
            if len(choix_col) > 1:
                df_ml = data[choix_col]
                df_ml = df_ml.dropna(axis=0)
                if len(df_ml)==0:
                    with col1:
                        st.write("##")
                        st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else :
                    with col1 :
                        # encodage !
                        col_to_encodage = st.multiselect("Selectionner les colonnes à encoder",["Toutes les colonnes"] + choix_col)
                        for col in col_to_encodage :
                            st.write("encodage colonne "+col+" : "+str(df_ml[col].unique().tolist())+"->"+str(np.arange(len(df_ml[col].unique()))))
                            df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())), inplace=True)  # encodage
                        ## on choisit notre modèle
                        from sklearn.neighbors import KNeighborsClassifier
                        model = KNeighborsClassifier()
                        ## création des target et features à partir du dataset
                        target = st.selectbox("Target :", ["Selectionner une target"] + col_numeric(df_ml))
                        with col2 :
                            if target != "Selectionner une target" :
                                y = df_ml[target]  # target
                                X = df_ml.drop(target, axis=1)  # features
                                try :
                                    model.fit(X, y)  # on entraine le modèle
                                    model.score(X, y)  # pourcentage de réussite

                                    model.predict(X)  # on test avec X

                                    features = []
                                    st.write("##")
                                    st.markdown('<p class="section">Entrez vos données</p>', unsafe_allow_html=True)
                                    for col in X.columns.tolist() :
                                        col = st.text_input(col, "")
                                        features.append(col)

                                    if "" not in features :
                                        x = np.array(features).reshape(1, len(features))
                                        p = (model.predict(x))
                                        st.write("##")
                                        st.success("Prédiction de la target "+target+" : "+str(p))
                                except :
                                    with col1:
                                        st.write("##")
                                        st.error("Erreur ! Avez vous encoder toutes les features necessaires ?")
    else :
        st.warning('Veuillez charger un dataset !')
### Fin section ML ###



if __name__=="__main__":
    main()

