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
from streamlit.hashing import _CodeHasher
from sklearn.decomposition import PCA
from umap import UMAP
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import tensorflow as tf
import PIL.Image
import tensorflow_hub as hub
try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

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
    border: solid #F65323 6px;
    padding: 5px;
}
.intro{
    text-align: justify;
    font-size:20px !important;
}
.grand_titre {
    font-size:30px !important;
    font-weight: bold;
    text-decoration: underline;
    text-decoration-color: #2782CD;
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
    text-decoration: underline;
    text-decoration-color: #258813;
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
</style>
""", unsafe_allow_html=True)


###### functions ########
def max_std(dataset):# colonne de maximum de variance
    l = []
    for nom in dataset.columns:
        if type(dataset[nom][0]) != object and type(dataset[nom][0]) != str:
            l.append([dataset[nom].std(), nom])
    return (max(l))

def col_numeric(df):#retourne les colonnes numériques d'un dataframe
    return df.select_dtypes(include=np.number).columns.tolist()

def clean_data(x):# enlever les symboles d'une colonne
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', '').replace('€', '').replace('£', ''))
    return(x)

def distance_e(x, y):  # distance entre 2 points du plan cartésien
    return distance.euclidean([x[0],x[1]],[y[0],y[1]])

def max_dist(donnee_apres_pca, df, voisins): # pour knn, retourne la distance du voisins le plus loin
    distances = []
    for i in range(len(df)):
        distances.append(distance_e(donnee_apres_pca, [df['x'].iloc[i], df['y'].iloc[i]]))
    distances.sort()
    return distances[voisins-1]

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

###### Session data ######
class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)

def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session

def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


####### Streamlit home ######
st.cache()
uploaded_file = "Datasets/iris.csv"
file_details = {"FileName" : 'csv'}
#uploaded_file = st.sidebar.file_uploader("Chargez votre dataset 📚", type=['csv', 'xls'])
#if uploaded_file is not None:
#    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type,"FileSize": uploaded_file.size}
#    st.sidebar.write(uploaded_file)
#    st.sidebar.success('Fichier chargé avec succès !')


#####################
### Main function ###
#####################
def main():
    state = _get_state()
    PAGES = {
        "Accueil": page1,
        "Chargement du dataset": page2,
        "Analyse des colonnes": page3,
        "Matrice de corrélation": page4,
        "Graphiques et Regressions" : page5,
        "Machine Learning": page6,
        "Deep Learning" : page7
    }
    st.sidebar.title('Menu :bulb:')
    page = st.sidebar.radio("", list(PAGES.keys()))
    PAGES[page](state)
    if state.data is not None:
        state.clear()
    state.sync()









###############
### Accueil ###
###############
def page1(state):
    st.markdown('<p class="first_titre">Preprocessing automatique</p>', unsafe_allow_html=True)
    st.write("##")
    st.markdown(
        '<p class="intro">Bienvenue sur le site de Preprocessing en ligne ! Déposez vos datasets csv et excel et commencez votre analyse dès maintenant ! Cherchez les variables les plus intéressantes, visualisez vos données et créez vos modèles de Machine et Deep Learning. ' +
        'Pour charger votre dataset, uploadé le depuis le volet latéral, et rendez vous dans la section "chargement du dataset".</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="intro">Un tutoriel sur l\'utilisation de ce site est disponible sur <a href="https://github.com/antonin-lfv/Online_preprocessing_for_ML">Github</a>. Si vous souhaitez un dataset pour ' +
        'simplement tester, vous pouvez télécharger le dataset des iris <a href="https://www.kaggle.com/arshid/iris-flower-dataset">ici</a>.</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="intro">En cas de bug ou d\'erreur veuillez m\'en informer par mail ou sur Discord. (Liens sur Github)</p>',
        unsafe_allow_html=True)
    st.write("##")
### Fin accueil ###










##########################
### section du dataset ###
##########################
def page2(state):
    st.markdown('<p class="grand_titre">Chargement du dataset</p>', unsafe_allow_html=True)

    col1_1, b_1, col2_1 = st.beta_columns((1, 0.1, 1))
    col1, b, col2 = st.beta_columns((2.7, 0.3, 1))
    if state.data is not None :
        with col1_1:
            state.separateur = st.text_input("Séparateur (optionnel): ", state.separateur or "")
        st.write("##")
        st.markdown("<p class='petite_section'>Modifications du dataset : </p>",unsafe_allow_html=True)
        col1_1, b_1, col2_1, c_1, col3_1 = st.beta_columns((1, 0.2, 1, 0.2, 1))  # pour time series
        st.write("##")
        with col1_1:
            state.col_to_time = st.multiselect('Conversion Time Series',
                                                state.data.columns.tolist(),
                                               state.col_to_time)
        with col2_1:
            state.col_to_float_money = st.multiselect('Conversion Monnaies',
                                                state.data.columns.tolist() ,
                                                      state.col_to_float_money)
        with col3_1:
            state.col_to_float_coma = st.multiselect('Conversion string avec virgules vers float',
                                                state.data.columns.tolist(),
                                                     state.col_to_float_coma)
        if len(state.col_to_time)>0 :
            with col1_1:
                for col in state.col_to_time:
                    try:
                        state.data[col] = pd.to_datetime(state.data[col])
                        st.success("Transformation effectuée !")
                    except:
                        st.error("Transformation impossible ou déjà effectuée")
        if len(state.col_to_float_money)>0 :
            with col2_1:
                for col in state.col_to_float_money:
                    try:
                        state.data[col] = state.data[col].apply(clean_data).astype('float')
                        st.success("Transformation effectuée !")
                    except:
                        st.error("Transformation impossible ou déjà effectuée")
        if len(state.col_to_float_coma)>0:
            with col3_1:
                for col in state.col_to_float_coma:
                    try:
                        state.data[col] = state.data[col].apply(lambda x: float(str(x).replace(',', '.')))
                        st.success("Transformation effectuée !")
                    except:
                        st.error("Transformation impossible ou déjà effectuée")
        with col1 :
            st.write("##")
            st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
            st.write(state.data.head(50))
            st.write("##")

        with col2 :
            st.write("##")
            st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
            st.write(' - Taille:', state.data.shape)
            st.write(' - Nombre de valeurs:', state.data.shape[0] * state.data.shape[1])
            st.write(' - Type des colonnes:', state.data.dtypes.value_counts())
            st.write(' - Pourcentage de valeurs manquantes:', round(
                sum(pd.DataFrame(state.data).isnull().sum(axis=1).tolist()) * 100 / (state.data.shape[0] * state.data.shape[1]), 2),
                     ' % (', sum(pd.DataFrame(state.data).isnull().sum(axis=1).tolist()), ')')

    if state.data is None:
        try:
            if 'csv' in file_details['FileName']:
                if state.separateur is not None:
                    data = pd.read_csv(uploaded_file, sep=state.separateur, engine='python')
                    state.data = data
                else :
                    data = pd.read_csv(uploaded_file, engine='python')
                    state.data = data
            else:
                if state.separateur is not None :
                    data = pd.read_excel(uploaded_file, sep=state.separateur, engine='python')
                    state.data = data
                else :
                    data = pd.read_excel(uploaded_file, engine='python')
                    state.data = data
        except:
            st.warning('Veuillez charger votre dataset')
### Fin section du dataset ###









#############################
### Section de la colonne ###
#############################
def page3(state):
    st.markdown('<p class="grand_titre">Analyse des colonnes</p>', unsafe_allow_html=True)
    st.write('##')
    if state.data is not None:
        options = state.data.columns.to_list()
        state.slider_col = st.multiselect(
            'Selectionner une ou plusieurs colonnes',
            options,
            state.slider_col
        )
        if state.slider_col :
            col1, b, col2, c = st.beta_columns((1.1, 0.1, 1.1, 0.3))
            with col1:
                st.write('##')
                st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
            with col2:
                st.write('##')
                st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
            for col in state.slider_col :
                ### Données ###
                data_col = state.data[col].copy()
                n_data = state.data[col].to_numpy()

                st.write('##')
                col1, b, col2, c = st.beta_columns((1,1,2, 0.5))
                with col1 :
                    st.markdown('<p class="nom_colonne_page3">'+col+'</p>', unsafe_allow_html=True)
                    st.write(data_col.head(20))
                with col2 :
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

                    st.write(' ● Nombre de valeurs manquantes:', sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist()))

                    st.write(' ● Longueur:', n_data.shape[0])

                    st.write(' ● Nombre de valeurs différentes non NaN:',
                             abs(len(Counter(n_data)) - sum(pd.DataFrame(n_data).isnull().sum(axis=1).tolist())))
                    ### Fin section données ###
                st.write('##')
    else :
        st.warning('Rendez-vous dans la section Chargement du dataset pour importer votre dataset')
### Fin section colonne ###









##########################
###Section Mat de corr ###
##########################
def page4(state):
    st.markdown('<p class="grand_titre">Matrice de corrélations</p>', unsafe_allow_html=True)
    st.write("##")
    if state.data is not None:
            col1, b, col2 = st.beta_columns((1, 1, 2))
            df_sans_NaN = state.data
            with col1:
                state.couleur_corr = st.selectbox('Couleur', ['Selectionner une colonne'] + df_sans_NaN.columns.tolist(), (['Selectionner une colonne'] + df_sans_NaN.columns.tolist()).index(state.couleur_corr) if state.couleur_corr else 0)
                st.write("##")
            state.select_columns_corr = st.multiselect("Choisir au moins deux colonnes",[ "Toutes les colonnes"] + col_numeric(df_sans_NaN), state.select_columns_corr)
            if len(state.select_columns_corr)>1 and "Toutes les colonnes" not in state.select_columns_corr:
                df_sans_NaN = pd.concat([state.data[col] for col in state.select_columns_corr],axis=1).dropna()
                if len(df_sans_NaN) == 0:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else :
                    if state.couleur_corr!='Selectionner une colonne':
                        fig=px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN[state.select_columns_corr]), color=state.couleur_corr, color_continuous_scale='Bluered_r')
                    else :
                        fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN[state.select_columns_corr]))
                    fig.update_layout(width=900, height=700,margin=dict(l=40, r=50, b=40, t=40), font=dict(size=7))
                    fig.update_layout({"xaxis" + str(i+1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN[state.select_columns_corr])))})
                    fig.update_layout({"yaxis" + str(i+1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN[state.select_columns_corr])))})
                    fig.update_traces(marker=dict(size=2))
                    fig.update_traces(diagonal_visible=False)
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig)
            elif state.select_columns_corr==["Toutes les colonnes"]:
                df_sans_NaN = state.data.dropna()
                if len(df_sans_NaN) == 0:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else :
                    if state.couleur_corr!='Selectionner une colonne':
                        fig=px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN), color=state.couleur_corr)
                    else :
                        fig = px.scatter_matrix(df_sans_NaN, dimensions=col_numeric(df_sans_NaN))
                    fig.update_layout({"xaxis" + str(i+1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
                    fig.update_layout({"yaxis" + str(i+1): dict(showticklabels=False) for i in range(len(col_numeric(df_sans_NaN)))})
                    fig.update_traces(marker=dict(size=2))
                    fig.update_layout(width=900, height=700,margin=dict(l=40, r=50, b=40, t=40),font=dict(size=7))
                    fig.update_traces(marker=dict(size=2))
                    fig.update_traces(diagonal_visible=False)
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig)
            elif len(state.select_columns_corr)>1 and "Toutes les colonnes" in state.select_columns_corr :
                st.error("Erreur de saisi !")
            else :
                pass
    else :
        st.warning('Rendez-vous dans la section Chargement du dataset pour importer votre dataset')
### Fin section graphiques ###











###########################
###  Section Graphiques ###
###########################
def page5(state):
    st.markdown('<p class="grand_titre">Graphiques et regressions</p>', unsafe_allow_html=True)
    st.write("##")
    if state.data is not None:
        col1, b, col2, c, col3, d, col4 = st.beta_columns((7)) # pour les autres select
        col_num = col_numeric(state.data)+state.col_to_time
        with col1 :
            st.write("##")
            state.abscisse_plot = st.selectbox('Données en abscisses', col_num,  col_num.index(state.abscisse_plot) if state.abscisse_plot else 0)
            state.ordonnee_plot = st.selectbox('Données en ordonnées', col_num, col_num.index(state.ordonnee_plot) if state.ordonnee_plot else 1)
            # couleur_plot = st.selectbox('Couleur', ['Selectionner une colonne'] + data.columns.tolist())
        with col2 :
            st.write("##")
            state.type_plot = st.radio("Type de plot", ['Points', 'Courbe', 'Latitude/Longitude', 'Histogramme'], ['Points', 'Courbe', 'Latitude/Longitude', 'Histogramme'].index(state.type_plot) if state.type_plot else 0)
            type_plot_dict = {
                'Courbe': 'lines',
                'Points': 'markers',
                'Latitude/Longitude': 'map',
            }
        st.write('##')
        if state.abscisse_plot and state.ordonnee_plot :
            if state.type_plot == 'Latitude/Longitude':
                fig = go.Figure()
                df_sans_NaN = pd.concat([state.data[state.abscisse_plot], state.data[state.ordonnee_plot]], axis=1).dropna()
                if len(df_sans_NaN)==0 :
                    st.warning('Le dataset composé des 2 colonnes selectionnées après dropna() est vide !')
                else :
                    fig.add_scattermapbox(
                        mode="markers",
                        lon=df_sans_NaN[state.ordonnee_plot],
                        lat=df_sans_NaN[state.abscisse_plot],
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
            elif state.type_plot=='Histogramme':
                fig=go.Figure()
                df_sans_NaN = pd.concat([state.data[state.abscisse_plot], state.data[state.ordonnee_plot]], axis=1).dropna()
                if len(df_sans_NaN)==0 :
                    st.warning('Le dataset composé des 2 colonnes selectionnées après dropna() est vide !')
                else :
                    fig.add_histogram(x=df_sans_NaN[state.abscisse_plot], y=df_sans_NaN[state.ordonnee_plot])
            else:
                with col3:
                    st.write("##")
                    st.write("##")
                    state.maximum = st.checkbox("Maximum", state.maximum)
                    state.moyenne = st.checkbox("Moyenne", state.moyenne)
                    state.minimum = st.checkbox("Minimum", state.minimum)
                fig = go.Figure()
                df_sans_NaN = pd.concat([state.data[state.abscisse_plot], state.data[state.ordonnee_plot]], axis=1).dropna()
                if len(df_sans_NaN)==0 :
                    st.warning('Le dataset composé des 2 colonnes selectionnées après dropna() est vide !')
                else :
                    fig.add_scatter(x=df_sans_NaN[state.abscisse_plot], y=df_sans_NaN[state.ordonnee_plot],mode=type_plot_dict[state.type_plot], name='', showlegend=False)
                    #if abscisse_plot not in col_to_time and ordonnee_plot not in col_to_time :
                    with col4:
                        st.write("##")
                        if state.type_plot == 'Points' or state.type_plot == 'Courbe' :
                            if state.abscisse_plot not in state.col_to_time and state.ordonnee_plot not in state.col_to_time:
                                st.write("##")
                                state.trendline = st.checkbox("Regression linéaire", state.trendline)
                                state.polynom_feat = st.checkbox("Regression polynomiale", state.polynom_feat)
                                if state.polynom_feat:
                                    state.degres = st.slider('Degres de la regression polynomiale', min_value=2,
                                                       max_value=100, value=state.degres)
                    if state.trendline :
                        # regression linaire
                        X = df_sans_NaN[state.abscisse_plot].values.reshape(-1, 1)
                        model = LinearRegression()
                        model.fit(X, df_sans_NaN[state.ordonnee_plot])
                        x_range = np.linspace(X.min(), X.max(), len(df_sans_NaN[state.ordonnee_plot]))
                        y_range = model.predict(x_range.reshape(-1, 1))
                        fig.add_scatter(x=x_range, y=y_range, name='Regression linéaire', mode='lines', marker=dict(color='red'))
                        # #################
                    if state.polynom_feat :
                        # regression polynomiale
                        X = df_sans_NaN[state.abscisse_plot].values.reshape(-1, 1)
                        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                        poly = PolynomialFeatures(state.degres)
                        poly.fit(X)
                        X_poly = poly.transform(X)
                        x_range_poly = poly.transform(x_range)
                        model = LinearRegression(fit_intercept=False)
                        model.fit(X_poly, df_sans_NaN[state.ordonnee_plot])
                        y_poly = model.predict(x_range_poly)
                        fig.add_scatter(x=x_range.squeeze(), y=y_poly, name='Polynomial Features', marker=dict(color='green'))
                        # #################
                    if state.moyenne :
                        # Moyenne #
                        fig.add_hline(y=df_sans_NaN[state.ordonnee_plot].mean(),
                                      line_dash="dot",
                                      annotation_text="moyenne : {}".format(round(df_sans_NaN[state.ordonnee_plot].mean(), 1)),
                                      annotation_position="bottom left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
                        pass
                    if state.minimum :
                        # Minimum #
                        fig.add_hline(y=df_sans_NaN[state.ordonnee_plot].min(),
                                      line_dash="dot",
                                      annotation_text="minimum : {}".format(round(df_sans_NaN[state.ordonnee_plot].min(), 1)),
                                      annotation_position="bottom left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
                        pass
                    if state.maximum :
                        # Maximum #
                        fig.add_hline(y=df_sans_NaN[state.ordonnee_plot].max(),
                                      line_dash="dot",
                                      annotation_text="maximum : {}".format(round(df_sans_NaN[state.ordonnee_plot].max(), 1)),
                                      annotation_position="top left",
                                      line_width=2, line=dict(color='black'),
                                      annotation=dict(font_size=10))
                        # #################
                        pass
            if len(df_sans_NaN) != 0:
                fig.update_xaxes(title_text=state.abscisse_plot)
                fig.update_yaxes(title_text=state.ordonnee_plot)
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
        st.warning('Rendez-vous dans la section Chargement du dataset pour importer votre dataset')

### Fin section mat de corr ###











###########################
######## Section ML #######
###########################
def page6(state):
    st.markdown('<p class="first_titre">Machine Learning</p>', unsafe_allow_html=True)
    st.write("##")
    PAGES_ML = {
        "K-nearest neighbors": page1_ML,
        "K-Means": page2_ML,
        "Support Vector Machine": page3_ML,
        "PCA": page4_ML,
        "UMAP": page5_ML,
    }
    st.sidebar.subheader("Algorithmes :control_knobs:")
    state.page_ml = st.sidebar.radio("", list(PAGES_ML.keys()), list(PAGES_ML.keys()).index(state.page_ml) if state.page_ml else 0)
    PAGES_ML[state.page_ml](state)
### fin accueil ML ###



## ML pages ##
# KNN
def page1_ML(state):
    st.write("##")
    st.markdown('<p class="grand_titre">KNN : k-nearest neighbors</p>', unsafe_allow_html=True)
    if state.data is not None:
            col1, b, col2 = st.beta_columns((1,0.2,1))
            with col1 :
                st.write("##")
                st.markdown('<p class="section">Selection des colonnes pour le modèle (target+features)</p>', unsafe_allow_html=True)
                state.choix_col = st.multiselect("Choisir au moins deux colonnes", state.data.columns.tolist(), state.choix_col)
            if len(state.choix_col) > 1:
                df_ml = state.data[state.choix_col]
                df_ml = df_ml.dropna(axis=0)
                if len(df_ml)==0:
                    with col1:
                        st.write("##")
                        st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else :
                    with col1 :
                        # encodage !
                        df_origine=df_ml.copy()
                        state.col_to_encodage = st.multiselect("Selectionner les colonnes à encoder",state.choix_col, state.col_to_encodage)
                        for col in state.col_to_encodage :
                            st.write("encodage colonne "+col+" : "+str(df_ml[col].unique().tolist())+"->"+str(np.arange(len(df_ml[col].unique()))))
                            df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())), inplace=True)  # encodage
                        ## création des target et features à partir du dataset
                        state.target = st.selectbox("Target :", ["Selectionner une target"] + col_numeric(df_ml), (["Selectionner une target"] + col_numeric(df_ml)).index(state.target) if state.target else 0 )
                        with col2 :
                            if state.target != "Selectionner une target" :
                                y = df_ml[state.target]  # target
                                X = df_ml.drop(state.target, axis=1)  # features
                                try :
                                    features = []
                                    st.write("##")
                                    st.markdown('<p class="section">Entrez vos données</p>', unsafe_allow_html=True)
                                    for col in X.columns.tolist() :
                                        col = st.text_input(col)
                                        features.append(col)

                                    if "" not in features:
                                        features = pd.DataFrame([features], columns=X.columns)  # données initiales
                                        X = X.append(features, ignore_index=True)

                                        ## PCA
                                        model = PCA(n_components=2)
                                        model.fit(X)
                                        x_pca = model.transform(X)
                                        df = pd.concat([pd.Series(x_pca[:-1, 0]).reset_index(drop=True), pd.Series(x_pca[:-1, 1]).reset_index(drop=True),pd.Series(df_origine[state.target]).reset_index(drop=True)], axis=1)
                                        df.columns = ["x", "y", str(state.target)]

                                        ## KNN
                                        with col1:
                                            st.write("##")
                                            st.write("##")
                                            st.markdown('<p class="section">Résultats</p>', unsafe_allow_html=True)
                                            state.voisins = st.slider('Nombre de voisins', min_value=4,max_value=int(len(y) * 0.2), value=state.voisins)
                                            y_pca_knn = df[state.target]  # target
                                            X_pca_knn = df.drop(state.target, axis=1)  # features
                                            model_knn = KNeighborsClassifier(n_neighbors=state.voisins)
                                            model_knn.fit(X_pca_knn, y_pca_knn)  # on entraine le modèle
                                            donnee_apres_pca = [x_pca[-1, 0], x_pca[-1, 1]]
                                            x = np.array(donnee_apres_pca).reshape(1, len(donnee_apres_pca))
                                            p = model_knn.predict(x)
                                            st.success("Prédiction de la target "+state.target+" : "+str(p))
                                            fig = px.scatter(df, x="x", y="y", color=str(state.target), labels={'color': str(state.target)}, color_discrete_sequence=px.colors.qualitative.Plotly)
                                            fig.update_layout(
                                                showlegend=True,
                                                template='simple_white',
                                                font=dict(size=10),
                                                autosize=False,
                                                width=1250, height=650,
                                                margin=dict(l=40, r=50, b=40, t=40),
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)',
                                                title="Prédiction avec " + str(state.voisins) + " voisins"
                                            )
                                            fig.update_yaxes(
                                                scaleanchor="x",
                                                scaleratio=1,
                                            )
                                            fig.add_scatter(x=[donnee_apres_pca[0]], y=[donnee_apres_pca[1]],
                                                            mode='markers', marker=dict(color='black'),
                                                            name='donnees pour prédiction')
                                            fig.add_shape(type="circle",
                                                          xref="x", yref="y",
                                                          x0=donnee_apres_pca[0] - max_dist(donnee_apres_pca, df, state.voisins),
                                                          y0=donnee_apres_pca[1] - max_dist(donnee_apres_pca, df, state.voisins),
                                                          x1=donnee_apres_pca[0] + max_dist(donnee_apres_pca, df, state.voisins),
                                                          y1=donnee_apres_pca[1] + max_dist(donnee_apres_pca, df, state.voisins),
                                                          line_color="red",
                                                          fillcolor="grey"
                                                          )
                                            fig.update(layout_coloraxis_showscale=False)
                                            with col1 :
                                                st.write("##")
                                                st.write("##")
                                                st.markdown('<p class="section">Visualisation grâce à une réduction de dimensions (PCA)</p>', unsafe_allow_html=True)
                                                st.write("##")
                                                st.plotly_chart(fig)
                                except :
                                    with col1:
                                        st.write("##")
                                        st.error("Erreur de chargement")
    else :
        st.warning('Rendez-vous dans la section Chargement du dataset pour importer votre dataset')

# K-Means
def page2_ML(state):
    st.write("##")
    st.markdown('<p class="grand_titre">K-Means</p>', unsafe_allow_html=True)
    if state.data is not None:
        col1, b, col2 = st.beta_columns((1, 0.2, 1))
        with col1:
            st.write("##")
            st.markdown('<p class="section">Selection des features pour le modèle</p>',unsafe_allow_html=True)
            state.choix_col_kmeans = st.multiselect("Choisir au moins deux colonnes", col_numeric(state.data), state.choix_col_kmeans)
        if len(state.choix_col_kmeans) > 1:
            df_ml = state.data[state.choix_col_kmeans]
            df_ml = df_ml.dropna(axis=0)
            if len(df_ml) == 0:
                with col1:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                with col1 :
                    X = df_ml[state.choix_col_kmeans]  # features
                    try:
                        ## PCA
                        model = PCA(n_components=2)
                        model.fit(X)
                        x_pca = model.transform(X)

                        df = pd.concat([pd.Series(x_pca[:, 0]), pd.Series(x_pca[:, 1])], axis=1)
                        df.columns = ["x", "y"]

                        ## K-Means
                        st.write("##")
                        st.markdown('<p class="section">Résultats</p>', unsafe_allow_html=True)
                        state.cluster = st.slider('Nombre de clusters', min_value=2,max_value=int(len(X) * 0.2), value=state.cluster)
                        X_pca_kmeans = df

                        modele = KMeans(n_clusters=state.cluster)
                        modele.fit(X_pca_kmeans)
                        y_kmeans = modele.predict(X_pca_kmeans)
                        df["class"] = pd.Series(y_kmeans)

                        fig = px.scatter(df, x=X_pca_kmeans['x'], y=X_pca_kmeans['y'], color="class", color_discrete_sequence=px.colors.qualitative.G10)
                        fig.update_layout(
                            showlegend=True,
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=1250, height=650,
                            margin=dict(l=40, r=50, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            title="K-Means avec " + str(state.cluster) + " Cluster",
                        )
                        fig.update(layout_coloraxis_showscale=False)
                        centers = modele.cluster_centers_
                        fig.add_scatter(x=centers[:, 0], y=centers[:, 1], mode='markers',marker=dict(color='black', size=15), opacity=0.4, name='Centroïdes')
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
        st.warning('Rendez-vous dans la section Chargement du dataset pour importer votre dataset')

# SVM
def page3_ML(state):
    st.write("##")
    st.markdown('<p class="grand_titre">SVM : Support Vector Machine</p>', unsafe_allow_html=True)
    if state.data is not None:
        st.write("##")
        st.markdown('<p class="section">Selection des features et de la target</p>', unsafe_allow_html=True)
        col1, b, col2 = st.beta_columns((1, 0.2, 1))
        with col1:
            state.choix_col_SVM = st.multiselect("Choisir deux colonnes", col_numeric(state.data),state.choix_col_SVM)
            state.choix_target_SVM = st.selectbox("Choisir la target", state.data.columns.tolist() , state.data.columns.tolist().index(state.choix_target_SVM) if state.choix_target_SVM else 0)

            if len(state.choix_col_SVM)==2 :
                target = state.choix_target_SVM
                features = state.choix_col_SVM

                # dataset avec features + target
                df = state.data[[target] + features]
                df.dropna(axis=0)

                if len(df) == 0:
                    with col1:
                        st.write("##")
                        st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
                else:
                    if state.choix_target_SVM in state.choix_col_SVM :
                        st.warning("La target ne doit pas appartenir aux features")
                    else :
                        if len(df[target].unique().tolist()) > 1 :
                            with col2 :
                                state.classes_SVM = st.multiselect("Choisir deux classes", df[state.choix_target_SVM].unique().tolist(), state.classes_SVM)
                                if len(state.classes_SVM) >1 :
                                    df = df.loc[(df[target] == state.classes_SVM[0]) | (df[target] == state.classes_SVM[1])]
                                    y = df[target]
                                    X = df[features]
                                    state.choix_kernel = st.selectbox("Choisir le type de noyau", ['Linéaire'], ['Linéaire'].index(state.choix_kernel) if state.choix_kernel else 0)

                                    if state.choix_kernel == 'Linéaire' :
                                        fig = px.scatter(df, x=features[0], y=features[1], color=target,
                                                         color_continuous_scale=px.colors.diverging.Picnic)
                                        fig.update(layout_coloraxis_showscale=False)

                                        from sklearn.svm import SVC  # "Support vector classifier"
                                        model = SVC(kernel='linear', C=1E10)
                                        model.fit(X, y)

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
                                        fig.add_scatter(x=xx, y=yy, line=dict(color='black', width=2), name='Hyperplan')

                                        # Hyperplans up et down
                                        b = model.support_vectors_[0]
                                        yy_down = a * xx + (b[1] - a * b[0])
                                        fig.add_scatter(x=xx, y=yy_down, line=dict(color='black', width=1, dash='dot'),
                                                        name='Marges')
                                        b = model.support_vectors_[-1]
                                        yy_up = a * xx + (b[1] - a * b[0])
                                        fig.add_scatter(x=xx, y=yy_up, line=dict(color='black', width=1, dash='dot'),
                                                        showlegend=False)
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
                                        with col1 :
                                            st.write("##")
                                            st.plotly_chart(fig)

                                elif len(state.classes_SVM) >2:
                                    st.warning("Saisie invalide - trop de colonne selectionnées")

                        else :
                            st.warning("Le dataset ne contient qu'une classe")
            elif len(state.choix_col_SVM)>2:
                st.warning("Saisie invalide - trop de colonne selectionnées")


    else:
        st.warning('Rendez-vous dans la section Chargement du dataset pour importer votre dataset')


# PCA
def page4_ML(state):
    st.write("##")
    st.markdown('<p class="grand_titre">PCA : Analyse en composantes principales</p>', unsafe_allow_html=True)
    if state.data is not None:
        col1, b, col2 = st.beta_columns((1, 0.2, 1))
        with col1:
            st.write("##")
            st.markdown('<p class="section">Selection des colonnes pour le modèle (target+features)</p>',unsafe_allow_html=True)
            state.choix_col_PCA = st.multiselect("Choisir au moins deux colonnes",state.data.columns.tolist(), state.choix_col_PCA)
        if len(state.choix_col_PCA) > 1:
            df_ml = state.data[state.choix_col_PCA]
            df_ml = df_ml.dropna(axis=0)
            state.df_ml_origine = df_ml.copy()
            if len(df_ml) == 0:
                with col1:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                with col1:
                    # encodage !
                    state.col_to_encodage_PCA = st.multiselect("Selectionner les colonnes à encoder",
                                                       state.choix_col_PCA,
                                                           state.col_to_encodage_PCA)
                    for col in state.col_to_encodage_PCA:
                        st.write("encodage colonne " + col + " : " + str(df_ml[col].unique().tolist()) + "->" + str(np.arange(len(df_ml[col].unique()))))
                        df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())),inplace=True)  # encodage
                    ## on choisit notre modèle
                    model = PCA(n_components=2)
                with col2:
                    ## création des target et features à partir du dataset
                    st.write("##")
                    st.write("##")
                    state.target_PCA = st.selectbox("Target :", ["Selectionner une target"] + col_numeric(df_ml),(["Selectionner une target"] + col_numeric(df_ml)).index(state.target_PCA) if state.target_PCA else 0)
                if state.target_PCA != "Selectionner une target":
                    y = df_ml[state.target_PCA]  # target
                    X = df_ml.drop(state.target_PCA, axis=1)  # features
                    try:
                        model.fit(X)
                        x_pca = model.transform(X)
                        st.write("##")
                        st.markdown('<p class="section">Résultats</p>', unsafe_allow_html=True)
                        # résultats points
                        state.df = pd.concat([pd.Series(x_pca[:, 0]), pd.Series(x_pca[:, 1]),pd.Series(state.df_ml_origine[state.target_PCA])], axis=1)
                        state.df.columns=["x", "y", str(state.target_PCA)]
                        fig=px.scatter(state.df, x="x", y="y", color=str(state.target_PCA), labels={'color':'{}'.format(str(state.target_PCA))}, color_discrete_sequence=px.colors.qualitative.Plotly)
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
        st.warning('Rendez-vous dans la section Chargement du dataset pour importer votre dataset')

# UMAP
def page5_ML(state):
    st.write("##")
    st.markdown('<p class="grand_titre">UMAP : Uniform Manifold Approximation and Projection</p>', unsafe_allow_html=True)
    if state.data is not None:
        col1, b, col2 = st.beta_columns((1, 0.2, 1))
        with col1:
            st.write("##")
            st.markdown('<p class="section">Selection des colonnes pour le modèle (target+features)</p>',unsafe_allow_html=True)
            state.choix_col_UMAP = st.multiselect("Choisir au moins deux colonnes",state.data.columns.tolist(), state.choix_col_UMAP)
        if len(state.choix_col_UMAP) > 1:
            df_ml = state.data[state.choix_col_UMAP]
            df_ml = df_ml.dropna(axis=0)
            state.df_ml_origine = df_ml.copy()
            if len(df_ml) == 0:
                with col1:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                with col1:
                    # encodage !
                    state.col_to_encodage_UMAP = st.multiselect("Selectionner les colonnes à encoder",
                                                       state.choix_col_UMAP,
                                                           state.col_to_encodage_UMAP)
                    for col in state.col_to_encodage_UMAP:
                        st.write("encodage colonne " + col + " : " + str(df_ml[col].unique().tolist()) + "->" + str(np.arange(len(df_ml[col].unique()))))
                        df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())),inplace=True)  # encodage
                    ## on choisit notre modèle
                    model = UMAP(random_state=0)
                with col2:
                    ## création des target et features à partir du dataset
                    st.write("##")
                    st.write("##")
                    state.target_UMAP = st.selectbox("Target :", ["Selectionner une target"] + col_numeric(df_ml),(["Selectionner une target"] + col_numeric(df_ml)).index(state.target_UMAP) if state.target_UMAP else 0)
                if state.target_UMAP != "Selectionner une target":
                    y = df_ml[state.target_UMAP]  # target
                    X = df_ml.drop(state.target_UMAP, axis=1)  # features
                    try:
                        model.fit(X)
                        x_umap = model.transform(X)
                        st.write("##")
                        st.markdown('<p class="section">Résultats</p>', unsafe_allow_html=True)
                        # résultats points
                        state.df = pd.concat([pd.Series(x_umap[:, 0]), pd.Series(x_umap[:, 1]),pd.Series(state.df_ml_origine[state.target_UMAP])], axis=1)
                        state.df.columns=["x", "y", str(state.target_UMAP)]
                        fig=px.scatter(state.df, x="x", y="y", color=str(state.target_UMAP), labels={'color':'{}'.format(str(state.target_UMAP))}, color_discrete_sequence=px.colors.qualitative.Plotly)
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
        st.warning('Rendez-vous dans la section Chargement du dataset pour importer votre dataset')
## Fin ML pages ##











###########################
########    D L     #######
###########################
def page7(state):
    st.markdown('<p class="first_titre">Deep Learning</p>', unsafe_allow_html=True)
    st.write("##")
    PAGES_DL = {
        "Transfert de style neuronal": page1_DL,
        "GAN": page2_DL,
    }

    st.sidebar.subheader("Algorithmes :control_knobs:")
    state.page_dl = st.sidebar.radio("", list(PAGES_DL.keys()), list(PAGES_DL.keys()).index(state.page_dl) if state.page_dl else 0)
    PAGES_DL[state.page_dl](state)
### Fin section DL ###



## DL pages ##
def page1_DL(state):
    st.write("##")
    st.markdown('<p class="grand_titre">Transfert de style neuronal</p>', unsafe_allow_html=True)
    st.write("##")
    content_path = {'Chat' : 'images/tensorflow_images/chat1.jpg',
                    'Los Angeles street':'images/tensorflow_images/LA_street.jpg'}
    style_path = {'La nuit étoilée - Van_Gogh' : 'images/tensorflow_images/Van_Gogh1.jpg',
                  'Guernica - Picasso' : 'images/tensorflow_images/GUERNICA.jpg',}
    col1, b, col2 = st.beta_columns((1, 0.2, 1))
    with col1:
        st.markdown('<p class="section">Selectionner une image de contenu</p>',unsafe_allow_html=True)
        state.image_contenu = st.selectbox("Choisir une image", list(content_path.keys()),list(content_path.keys()).index(state.image_contenu) if state.image_contenu else 0)
        content_image = load_img(content_path[state.image_contenu])
        content_image_plot = tf.squeeze(content_image, axis=0)
        fig = px.imshow(content_image_plot)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(
            showlegend=False,
            font=dict(size=10),
            width=600, height=300,
            margin=dict(l=40, r=50, b=40, t=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig)
    with col2:
        st.markdown('<p class="section">Selectionner une image de style</p>', unsafe_allow_html=True)
        state.image_style = st.selectbox("Choisir une image", list(style_path.keys()),list(style_path.keys()).index(state.image_style) if state.image_style else 0)
        style_image = load_img(style_path[state.image_style])
        style_image_plot = tf.squeeze(style_image, axis=0)
        fig = px.imshow(style_image_plot)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(
            showlegend=False,
            font=dict(size=10),
            width=600, height=300,
            margin=dict(l=40, r=50, b=40, t=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig)
    if st.button("Lancer le transfert"):
        st.write("##")
        st.markdown('<p class="section">Résultat</p>', unsafe_allow_html=True)
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        img = tensor_to_image(stylized_image)
        fig = px.imshow(img)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(
            showlegend=False,
            font=dict(size=10),
            width=1300, height=600,
            margin=dict(l=40, r=50, b=40, t=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig)


def page2_DL(state):
    st.write("##")
    st.markdown('<p class="grand_titre">GAN : Generative adversarial network</p>', unsafe_allow_html=True)
    if state.data is not None:
        st.write("##")
        st.write("Section en cours de developpement")
    else:
        st.warning('Rendez-vous dans la section Chargement du dataset pour importer votre dataset')
## Fin DL pages ##






if __name__=="__main__":
    main()

