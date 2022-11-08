# ===== Importations ===== #
import streamlit as st
from utils import *
# ===== html/css config ===== #
st.set_page_config(layout="wide", page_title="Section graphiques")
st.markdown(CSS, unsafe_allow_html=True)

# ===== Session ===== #
if "degres" not in st.session_state:
    st.session_state.degres = ""

# ===== Page ===== #
st.markdown('<p class="grand_titre">Graphiques</p>', unsafe_allow_html=True)
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
            st.session_state.type_plot = st.radio(label="label",
                                                    options=['Points', 'Courbe', 'Latitude/Longitude', 'Histogramme'],
                                                    help="Choisissez le type qui vous convient",
                                                    label_visibility='hidden'
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
                st.plotly_chart(fig, use_container_width=True)
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
                            with st.expander("Régressions rapides"):
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
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Veuillez charger vos données dans la section Dataset")
    st.write("##")
    st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)