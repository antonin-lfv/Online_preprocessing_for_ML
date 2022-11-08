# ===== Importations ===== #
import streamlit as st
from utils import *
# ===== html/css config ===== #
st.set_page_config(layout="wide", 
    page_title="No code AI", 
    menu_items={'About': "No-code AI Platform - réalisé par Antonin"},
)
st.markdown(CSS, unsafe_allow_html=True)

# ===== Session ===== #
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
if "file_details" not in st.session_state:
    st.session_state.file_details = ""

# ===== Page ===== #
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
c1, _, c2, _, _, _ = st.columns(6)
with c1:
    st.subheader("Liens")
    st.write(
        "• [Mon profil GitHub](https://github.com/antonin-lfv/Online_preprocessing_for_ML/blob/master/README.md)")
    st.write("• [Mon site](https://antonin-lfv.github.io)")
with c2:
    lottie_accueil = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_xRmNN8.json')
    st_lottie(lottie_accueil, height=200)
