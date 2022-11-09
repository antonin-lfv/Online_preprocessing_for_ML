# ===== Importations ===== #
import streamlit as st
from utils import *
# ===== html/css config ===== #
st.set_page_config(layout="wide", page_title="Régressions")
st.markdown(CSS, unsafe_allow_html=True)

# ===== Session ===== #
if "choix_features_reg" not in st.session_state:
    st.session_state.choix_features_reg = ""

# ===== Page ===== #
st.markdown('<p class="grand_titre">Régressions</p>', unsafe_allow_html=True)
st.write("##")
exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
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
    _, box1, _ = st.columns((0.2, 1, 0.1))
    _, box2_title, _ = st.columns((0.1, 1, 0.1))
    _, box2_eval1, box2_eval2, box2_eval3, box2_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
    _, box2, _ = st.columns((0.2, 1, 0.1))
    _, box3_title, _ = st.columns((0.1, 1, 0.1))
    _, box3_eval1, box3_eval2, box3_eval3, box3_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
    _, box3, _ = st.columns((0.2, 1, 0.1))
    _, box4_title, _ = st.columns((0.1, 1, 0.1))
    _, box4_eval1, box4_eval2, box4_eval3, box4_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
    _, box4, _ = st.columns((0.2, 1, 0.1))
    _, box5_title, _ = st.columns((0.1, 1, 0.1))
    _, box5_eval1, box5_eval2, box5_eval3, box5_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
    _, box5, _ = st.columns((0.2, 1, 0.1))
    _, box6_title, _ = st.columns((0.1, 1, 0.1))
    _, box6_eval1, box6_eval2, box6_eval3, box6_eval4, _ = st.columns((0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
    _, box6, _ = st.columns((0.2, 1, 0.1))
    with exp2:
        st.write("##")
        st.markdown('<p class="section">Selection des données pour la régression</p>', unsafe_allow_html=True)
    with col1_abscisse_reg:
        st.session_state.choix_features_reg = st.multiselect("Features",
                                                                col_numeric(st.session_state.data),
                                                                
                                                                )
    with col1_ordonnee_reg:
        st.session_state.choix_target_reg = st.selectbox("Target",
                                                            col_numeric(st.session_state.data)[::-1],
                                                            )
        st.write("##")
    if st.session_state.choix_target_reg in st.session_state.choix_features_reg:
        with warning1:
            st.warning("La target ne doit pas appartenir aux features")
    elif len(st.session_state.choix_features_reg) > 0:
        # Chargement des données - On enlève les valeurs manquantes
        df_sans_NaN = pd.concat(
            [st.session_state.data[st.session_state.choix_features_reg].reset_index(drop=True),
                st.session_state.data[st.session_state.choix_target_reg].reset_index(drop=True)],
            axis=1).dropna()
        if len(df_sans_NaN) == 0:
            with exp2:
                st.warning('Le dataset composé des 2 colonnes selectionnées après dropna() est vide !')

        else:  # Le dataset est viable
            try:
                # Data
                X_train, X_test, y_train, y_test = train_test_split(
                    df_sans_NaN[st.session_state.choix_features_reg].values,
                    df_sans_NaN[st.session_state.choix_target_reg], test_size=0.4, random_state=4)

                X_train, X_test, y_train, y_test = scale(X_train), scale(X_test), scale(y_train), scale(y_test)
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
                with box1_eval1:
                    st.metric(label="MSE (par rapport au train)", value=round(MSE_reg_test, 3),
                                delta=round(MSE_reg_test - MSE_reg_train, 3), delta_color="inverse")
                with box1_eval2:
                    st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                delta=round(RMSE_reg_test - RMSE_reg_train, 3), delta_color="inverse")
                with box1_eval3:
                    st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                delta=round(MAE_reg_test - MAE_reg_train, 3), delta_color="inverse")
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
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("##")
                    with st.expander("Code"):
                        st.code(streamlit_code_regression(st.session_state.choix_features_reg, st.session_state.choix_target_reg, model))
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
                                delta=round(MSE_reg_test - MSE_reg_train, 3), delta_color="inverse")
                with box2_eval2:
                    st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                delta=round(RMSE_reg_test - RMSE_reg_train, 3), delta_color="inverse")
                with box2_eval3:
                    st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                delta=round(MAE_reg_test - MAE_reg_train, 3), delta_color="inverse")
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
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("##")
                    with st.expander("Code"):
                        st.code(streamlit_code_regression(st.session_state.choix_features_reg, st.session_state.choix_target_reg, polynomial=True, model1=model1, model2=model2))
                    st.write('---')
                    st.write("##")

                # ###############################################################################
                if np.issubdtype(y_train.dtype, int) and np.any(y_train < 0):
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
                                    delta=round(MSE_reg_test - MSE_reg_train, 3), delta_color="inverse")
                    with box3_eval2:
                        st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                    delta=round(RMSE_reg_test - RMSE_reg_train, 3), delta_color="inverse")
                    with box3_eval3:
                        st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                    delta=round(MAE_reg_test - MAE_reg_train, 3), delta_color="inverse")
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
                        st.plotly_chart(fig, use_container_width=True)
                        st.write("##")
                        with st.expander("Code"):
                            st.code(streamlit_code_regression(st.session_state.choix_features_reg, st.session_state.choix_target_reg, model))
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
                                delta=round(MSE_reg_test - MSE_reg_train, 3), delta_color="inverse")
                with box4_eval2:
                    st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                delta=round(RMSE_reg_test - RMSE_reg_train, 3), delta_color="inverse")
                with box4_eval3:
                    st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                delta=round(MAE_reg_test - MAE_reg_train, 3), delta_color="inverse")
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
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("##")
                    with st.expander("Code"):
                        st.code(streamlit_code_regression(st.session_state.choix_features_reg, st.session_state.choix_target_reg, model))
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
                                delta=round(MSE_reg_test - MSE_reg_train, 3), delta_color="inverse")
                with box5_eval2:
                    st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                delta=round(RMSE_reg_test - RMSE_reg_train, 3), delta_color="inverse")
                with box5_eval3:
                    st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                delta=round(MAE_reg_test - MAE_reg_train, 3), delta_color="inverse")
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
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("##")
                    with st.expander("Code"):
                        st.code(streamlit_code_regression(st.session_state.choix_features_reg, st.session_state.choix_target_reg, model))
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
                                delta=round(MSE_reg_test - MSE_reg_train, 3), delta_color="inverse")
                with box6_eval2:
                    st.metric(label="RMSE (par rapport au train)", value=round(RMSE_reg_test, 3),
                                delta=round(RMSE_reg_test - RMSE_reg_train, 3), delta_color="inverse")
                with box6_eval3:
                    st.metric(label="MAE (par rapport au train)", value=round(MAE_reg_test, 3),
                                delta=round(MAE_reg_test - MAE_reg_train, 3), delta_color="inverse")
                with box6_eval4:
                    st.metric(label="r² (par rapport au train)", value=round(r2_reg_test, 3),
                                delta=round(r2_reg_test - r2_reg_train, 3), delta_color="inverse")
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
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("##")
                    with st.expander("Code"):
                        st.code(streamlit_code_regression(st.session_state.choix_features_reg, st.session_state.choix_target_reg, model))
                    st.write("##")

            except:
                with box2_title:
                    st.info("Impossible d'effectuer les régressions avec ces données")

else:
    with exp2:
        st.write("##")
        st.info('Rendez-vous dans la section Dataset pour importer votre dataset')
        st.write("##")
        st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)