# ===== Importations ===== #
import streamlit as st
from utils import *
# ===== html/css config ===== #
st.set_page_config(layout="wide", page_title="Analyse des colonnes")
st.markdown(CSS, unsafe_allow_html=True)

# ===== Session ===== #
if "slider_col" not in st.session_state:
    st.session_state.slider_col = ""

# ===== Page ===== #
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
    st.write("##")
    st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)