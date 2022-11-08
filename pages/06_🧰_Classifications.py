# ===== Importations ===== #
import streamlit as st
from utils import *
# ===== html/css config ===== #
st.set_page_config(layout="wide", page_title="Classifications")
st.markdown(CSS, unsafe_allow_html=True)

# ===== choix modèle ===== #
PAGES_classification = ["🪛 k-Nearest Neighbors", "🪛 k-Means", "🪛 Support Vector Machine", "🪛 Decision Tree", "🪛 Logistic regression", "🪛 DBSCAN"]
st.sidebar.selectbox(label="label", options=PAGES_classification, key="choix_page_classification", label_visibility='hidden')

# ===== Session ===== #
if "col_to_encodage_knn" not in st.session_state:
    st.session_state.col_to_encodage_knn = ""
if "choix_col_kmeans" not in st.session_state:
    st.session_state.choix_col_kmeans = ""
if "choix_col_SVM" not in st.session_state:
    st.session_state.choix_col_SVM = ""
if "classes_SVM" not in st.session_state:
    st.session_state.classes_SVM = ""
if "choix_col_DT" not in st.session_state:
    st.session_state.choix_col_DT = ""
if "choix_col_LR" not in st.session_state:
    st.session_state.choix_col_LR = ""
if "classes_LR" not in st.session_state:
    st.session_state.classes_LR = ""
if "choix_col_dbscan" not in st.session_state:
    st.session_state.choix_col_dbscan = ""

# ===== Page ===== #
if st.session_state.choix_page_classification == "🪛 k-Nearest Neighbors":
    st.markdown('<p class="grand_titre">KNN : k-nearest neighbors</p>', unsafe_allow_html=True)
    st.write("##")
    exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
    with exp2:
        with st.expander("Principe de l'algorithme KNN"):
            st.write("""
            * 1ère étape : Choix du nombre de voisins k
            * 2ème étape : Calcul de la distance entre le point non classifié et tous les autre
            * 3ème étape : Sélection des k plus proches voisins
            * 4ème étape : On compte le nombre de voisins dans chaque classe
            * 5ème étape : Attribution de la classe la plus présente à notre point 
            """)
    if 'data' in st.session_state:
        _, col1_features_encode, _ = st.columns((0.1, 1, 0.1))
        _, sub_col1, _ = st.columns((0.4, 0.5, 0.4))
        _, col1_target, _ = st.columns((0.1, 1, 0.1))
        _, col_best_score, _ = st.columns((0.4, 0.4, 0.4))
        _, col_titre_eval_model, _ = st.columns((0.4, 0.4, 0.4))
        _, col1_eval_modele, col2_eval_modele, col3_eval_modele, col4_eval_modele, _ = st.columns(
            (0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
        _, col1_roc, _ = st.columns((0.1, 1, 0.1))
        _, col1_AUC_value, _ = st.columns((0.4, 0.4, 0.4))
        _, col_learning_curve, _ = st.columns((0.1, 4, 0.1))
        _, col_data_to_predict, _ = st.columns((0.1, 0.75, 0.1))
        _, col_pca_knn, _ = st.columns((0.1, 1, 0.1))
        _, sub_col_prediction_knn, _ = st.columns((0.4, 0.75, 0.4))
        _, col_pca_knn_plot, _ = st.columns((0.1, 4, 0.1))
        with col1_features_encode:
            st.write("##")
            st.markdown('<p class="section">Selection des colonnes pour le modèle (target+features)</p>',
                        unsafe_allow_html=True)
            st.session_state.choix_col_knn = st.multiselect("Choisir au moins deux colonnes",
                                                        st.session_state.data.columns.tolist(),
                                                    
                                                        )
        if len(st.session_state.choix_col_knn) > 1:
            df_ml = st.session_state.data[st.session_state.choix_col_knn]
            df_ml = df_ml.dropna(axis=0)
            if len(df_ml) == 0:
                with col1_features_encode:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                # encodage !
                df_origine = df_ml.copy()
                with col1_features_encode:
                    st.session_state.col_to_encodage_knn = st.multiselect("Selectionner les colonnes à encoder",
                                                                        st.session_state.choix_col_knn,
                                                                        
                                                                        )
                with sub_col1:
                    with st.expander('Encodage'):
                        for col in st.session_state.col_to_encodage_knn:
                            st.write("Colonne " + col + "  :  " + str(df_ml[col].unique().tolist()) + " -> " + str(
                                np.arange(len(df_ml[col].unique()))))
                            df_ml[col].replace(df_ml[col].unique(), np.arange(len(df_ml[col].unique())),
                                                inplace=True)  # encodage
                ## création des target et features à partir du dataset
                with col1_target:
                    st.session_state.target_knn = st.selectbox("Target :",
                                                            ["Selectionner une target"] + col_numeric(df_ml),
                                                            )
                if st.session_state.target_knn != "Selectionner une target":
                    try:
                        ## KNN
                        st.write("##")
                        y_knn = df_ml[st.session_state.target_knn]  # target
                        X_knn = df_ml.drop(st.session_state.target_knn, axis=1)  # features
                        X_train, X_test, y_train, y_test = train_test_split(X_knn, y_knn,
                                                                            test_size=0.4,
                                                                            random_state=4)
                        # Gridsearchcv
                        params = {'n_neighbors': np.arange(1, 20)}
                        grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=4)
                        grid.fit(X_train.values, y_train.values)
                        best_k = grid.best_params_['n_neighbors']
                        best_model_knn = grid.best_estimator_
                        best_model_knn.fit(X_knn.values, y_knn.values)  # on entraine le modèle

                        # Meilleurs hyper params
                        with col_best_score:
                            st.write("##")
                            st.write("##")
                            st.markdown('<p class="section">Sélection des meilleurs hyper-paramètres</p>',
                                        unsafe_allow_html=True)
                            st.write("##")
                            st.success(
                                f'Après un GridSearchCV on prendra **k = {best_k}** voisins')
                            st.write("##")

                        # Évaluation du modèle
                        y_pred_test = best_model_knn.predict(X_test.values)
                        y_pred_train = best_model_knn.predict(X_train.values)
                        if len(y_knn.unique()) > 2:
                            with col_titre_eval_model:
                                st.write("##")
                                st.markdown(
                                    '<p class="section">Évaluation par rapport au train set</p>',
                                    unsafe_allow_html=True)
                                st.write("##")
                                # average='micro' car nos label contiennent plus de 2 classes
                                # Test set
                                precis_test = precision_score(y_test, y_pred_test, average='micro')
                                rappel_test = recall_score(y_test, y_pred_test, average='micro')
                                F1_test = f1_score(y_test, y_pred_test, average='micro')
                                accur_test = accuracy_score(y_test, y_pred_test)
                                # Train set
                                precis_train = precision_score(y_train, y_pred_train, average='micro')
                                rappel_train = recall_score(y_train, y_pred_train, average='micro')
                                F1_train = f1_score(y_train, y_pred_train, average='micro')
                                accur_train = accuracy_score(y_train, y_pred_train)
                                with col1_eval_modele:
                                    st.metric(label="Precision", value=round(precis_test, 3),
                                                delta=round(precis_test - precis_train, 3))
                                with col2_eval_modele:
                                    st.metric(label="Recall", value=round(rappel_test, 3),
                                                delta=round(rappel_test - rappel_train, 3))
                                with col3_eval_modele:
                                    st.metric(label="F1 score", value=round(F1_test, 3),
                                                delta=round(F1_test - F1_train, 3))
                                with col4_eval_modele:
                                    st.metric(label="Accuracy", value=round(accur_test, 3),
                                                delta=round(accur_test - accur_train, 3))

                        else:
                            with col_titre_eval_model:
                                st.write("##")
                                st.markdown(
                                    '<p class="section">Évaluation par rapport au train set</p>',
                                    unsafe_allow_html=True)
                                st.write("##")
                                # label binaire
                                # Test set
                                precis_test = precision_score(y_test, y_pred_test)
                                rappel_test = recall_score(y_test, y_pred_test)
                                F1_test = f1_score(y_test, y_pred_test)
                                accur_test = accuracy_score(y_test, y_pred_test)
                                # Train set
                                precis_train = precision_score(y_train, y_pred_train)
                                rappel_train = recall_score(y_train, y_pred_train)
                                F1_train = f1_score(y_train, y_pred_train)
                                accur_train = accuracy_score(y_train, y_pred_train)
                                with col1_eval_modele:
                                    st.metric(label="Precision", value=round(precis_test, 3),
                                                delta=round(precis_test - precis_train, 3))
                                with col2_eval_modele:
                                    st.metric(label="Recall", value=round(rappel_test, 3),
                                                delta=round(rappel_test - rappel_train, 3))
                                with col3_eval_modele:
                                    st.metric(label="F1 score", value=round(F1_test, 3),
                                                delta=round(F1_test - F1_train, 3))
                                with col4_eval_modele:
                                    st.metric(label="Accuracy", value=round(accur_test, 3),
                                                delta=round(accur_test - accur_train, 3))

                            with col1_roc:
                                st.write("##")
                                st.write("##")
                                st.markdown(
                                    '<p class="section">ROC curve</p>',
                                    unsafe_allow_html=True)
                                fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
                                with col1_AUC_value:
                                    st.write("##")
                                    st.info(f'Area Under the Curve (AUC) = {round(auc(fpr, tpr), 4)}')
                                fig = px.area(
                                    x=fpr, y=tpr,
                                    labels=dict(x='False Positive Rate', y='True Positive Rate'),
                                    width=500, height=500,
                                )
                                fig.add_shape(
                                    type='line', line=dict(dash='dash'),
                                    x0=0, x1=1, y0=0, y1=1
                                )

                                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                                fig.update_xaxes(constrain='domain')
                                fig.update_layout(
                                    font=dict(size=10),
                                    autosize=False,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    width=1050, height=650,
                                    margin=dict(l=40, r=50, b=40, t=40),
                                )
                                st.plotly_chart(fig, use_container_width=True)

                        # Learning curve
                        with col_learning_curve:
                            st.write("##")
                            st.markdown(
                                '<p class="section">Learning curves</p>',
                                unsafe_allow_html=True)
                            st.write("##")
                            N, train_score, val_score = learning_curve(best_model_knn, X_train, y_train,
                                                                        train_sizes=np.linspace(0.2,
                                                                                                1.0,
                                                                                                10),
                                                                        cv=3, random_state=4)
                            fig = go.Figure()
                            fig.add_scatter(x=N, y=train_score.mean(axis=1), name='train',
                                            marker=dict(color='deepskyblue'))
                            fig.add_scatter(x=N, y=val_score.mean(axis=1), name='validation',
                                            marker=dict(color='red'))
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
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(
                                "Il est possible que votre dataset soit trop petit pour effectuer la cross-validation dans de bonnes conditions")

                        # Faire une prédiction
                        with col_data_to_predict:
                            st.write("##")
                            st.write("##")
                            st.markdown('---')
                            st.write("##")
                            st.markdown(
                                '<p class="section">Prédiction à l\'aide du modèle</p>',
                                unsafe_allow_html=True)
                            st.write("##")
                            st.write("##")
                            features = []
                            st.markdown('<p class="section">Entrez vos données</p>', unsafe_allow_html=True)
                            st.write("##")
                            for col in X_test.columns.tolist():
                                col = st.text_input(col)
                                features.append(col)
                        if "" not in features:
                            prediction_knn = best_model_knn.predict(np.array(features, dtype=float).reshape(1, -1))
                            with sub_col_prediction_knn:
                                st.write("##")
                                st.success(
                                    f'Prédiction de la target {st.session_state.target_knn} avec les données entrées : **{str(df_origine[st.session_state.target_knn].unique()[int(prediction_knn[0])])}**')
                                st.write("##")
                    except:
                        with col_best_score:
                            st.write("##")
                            st.warning("Impossible d'utiliser ce modèle avec ces données")
    else:
        with exp2:
            st.write("##")
            st.info('Rendez-vous dans la section Dataset pour importer votre dataset')
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)

elif st.session_state.choix_page_classification == "🪛 k-Means":
    st.markdown('<p class="grand_titre">K-Means</p>', unsafe_allow_html=True)
    st.write("##")
    exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
    with exp2:
        with st.expander("Principe de l'algorithme des K-means"):
            st.write("""
            Le but est de diviser les points en k clusters.
            * 1ère étape : On place au hasard les k centroïdes
            * 2ème étape : Pour chaque point, on lui associe l'étiquette du centroïde le plus proche
            * 3ème étape : On calcul le centre de gravité des k clusters que nous venons de créer 
            * 4ème étape : On réitère les étapes 2 et 3 jusqu'à temps que les centroïdes ne bougent plus
            
            On peut utiliser diverses techniques pour choisir les premiers centroïdes, et diverses métriques 
            pour évaluer les distances.
            """)
    if 'data' in st.session_state:
        _, col1_features_choice, _ = st.columns((0.1, 1, 0.1))
        with col1_features_choice:
            st.write("##")
            st.markdown('<p class="section">Selection des features pour le modèle</p>', unsafe_allow_html=True)
            st.session_state.choix_col_kmeans = st.multiselect("Choisir au moins deux colonnes",
                                                                col_numeric(st.session_state.data),
                                                                
                                                                )
        if len(st.session_state.choix_col_kmeans) > 1:
            df_ml = st.session_state.data[st.session_state.choix_col_kmeans]
            df_ml = df_ml.dropna(axis=0)
            if len(df_ml) == 0:
                with col1_features_choice:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                with col1_features_choice:
                    X = df_ml[st.session_state.choix_col_kmeans]  # features
                    try:
                        ## PCA
                        model = PCA(n_components=2)
                        model.fit(X)
                        x_pca = model.transform(X)

                        df = pd.concat([pd.Series(x_pca[:, 0]).reset_index(drop=True),
                                        pd.Series(x_pca[:, 1]).reset_index(drop=True)], axis=1)
                        df.columns = ["x", "y"]

                        ## K-Means
                        st.write("##")
                        st.markdown('<p class="section">Résultats</p>', unsafe_allow_html=True)
                        st.session_state.cluster = st.slider('Nombre de clusters', min_value=2,
                                                                max_value=int(len(X) * 0.2),
                                                                )
                        X_pca_kmeans = df

                        modele = KMeans(n_clusters=st.session_state.cluster)
                        modele.fit(X_pca_kmeans)
                        y_kmeans = modele.predict(X_pca_kmeans)
                        df["class"] = pd.Series(y_kmeans)

                        fig = px.scatter(df, x=X_pca_kmeans['x'], y=X_pca_kmeans['y'], color="class",
                                            color_discrete_sequence=px.colors.qualitative.G10)
                        fig.update_layout(
                            showlegend=True,
                            template='simple_white',
                            font=dict(size=10),
                            autosize=False,
                            width=1000, height=650,
                            margin=dict(l=40, r=50, b=40, t=40),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            title="K-Means avec " + str(st.session_state.cluster) + " Cluster",
                        )
                        fig.update(layout_coloraxis_showscale=False)
                        centers = modele.cluster_centers_
                        fig.add_scatter(x=centers[:, 0], y=centers[:, 1], mode='markers',
                                        marker=dict(color='black', size=15), opacity=0.4, name='Centroïdes')
                        st.write("##")
                        st.markdown(
                            '<p class="section">Visualisation grâce à une réduction de dimensions (PCA)</p>',
                            unsafe_allow_html=True)
                        st.write("##")
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        with col1_features_choice:
                            st.write("##")
                            st.error("Erreur de chargement")
    else:
        with exp2:
            st.write("##")
            st.info('Rendez-vous dans la section Dataset pour importer votre dataset')
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)


elif st.session_state.choix_page_classification == "🪛 Support Vector Machine":
    st.markdown('<p class="grand_titre">SVM : Support Vector Machine</p>', unsafe_allow_html=True)
    st.write("##")
    exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
    with exp2:
        with st.expander("Principe des Support Vector Machine"):
            st.write("""
            Le but est de séparer les classes à l'aide d'une droite/courbe qui va maximiser sa distance avec les points les plus proches (les support vectors).
            
            On peut pour cela utiliser différents noyaux qui peuvent être linéaire ou polynomial.
            """)
    if 'data' in st.session_state:
        st.write("##")
        st.markdown('<p class="section">Selection des features et de la target</p>', unsafe_allow_html=True)
        _, col1_km, _ = st.columns((0.1, 1, 0.1))
        with col1_km:
            st.session_state.choix_col_SVM = st.multiselect("Choisir deux colonnes",
                                                            col_numeric(st.session_state.data),
                                                            help="Vos features", max_selections=2,
                                                            
                                                            )
            st.session_state.choix_target_SVM = st.selectbox("Choisir la target",
                                                                st.session_state.data.columns.tolist()[::-1],
                                                                )

        if len(st.session_state.choix_col_SVM) == 2:
            target = st.session_state.choix_target_SVM
            features = st.session_state.choix_col_SVM

            # dataset avec features + target
            df = st.session_state.data[[target] + features]
            df.dropna(axis=0, inplace=True)

            if len(df) == 0:
                with col1_km:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                if st.session_state.choix_target_SVM in st.session_state.choix_col_SVM:
                    with col1_km:
                        st.warning("La target ne doit pas appartenir aux features")
                else:
                    if len(df[target].unique().tolist()) > 1:
                        with col1_km:
                            st.session_state.classes_SVM = st.multiselect("Choisir deux classes",
                                                                            df[st.session_state.choix_target_SVM].unique().tolist(), 
                                                                                max_selections=2, )
                            if len(st.session_state.classes_SVM) > 1:
                                df = df.loc[
                                    (df[target] == st.session_state.classes_SVM[0]) | (
                                            df[target] == st.session_state.classes_SVM[1])]
                                y = df[target]
                                X = df[features]
                                st.session_state.choix_kernel = st.selectbox("Choisir le type de noyau",
                                                                                ['Linéaire'],
                                                                                )

                                if st.session_state.choix_kernel == 'Linéaire':
                                    fig = px.scatter(df, x=features[0], y=features[1], color=target,
                                                        color_continuous_scale=px.colors.diverging.Picnic)
                                    fig.update(layout_coloraxis_showscale=False)

                                    from sklearn.svm import SVC  # "Support vector classifier"

                                    model = SVC(kernel='linear', C=1E10)
                                    model.fit(X, y)  # to do ajouter un gridsearchcv

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
                                    fig.add_scatter(x=xx, y=yy, line=dict(color='black', width=2),
                                                    name='Hyperplan')

                                    # Hyperplans up et down
                                    b = model.support_vectors_[0]
                                    yy_down = a * xx + (b[1] - a * b[0])
                                    fig.add_scatter(x=xx, y=yy_down,
                                                    line=dict(color='black', width=1, dash='dot'),
                                                    name='Marges')
                                    b = model.support_vectors_[-1]
                                    yy_up = a * xx + (b[1] - a * b[0])
                                    fig.add_scatter(x=xx, y=yy_up,
                                                    line=dict(color='black', width=1, dash='dot'),
                                                    showlegend=False)
                                    fig.update_layout(
                                        showlegend=True,
                                        template='simple_white',
                                        font=dict(size=10),
                                        autosize=False,
                                        width=1000, height=650,
                                        margin=dict(l=40, r=50, b=40, t=40),
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                    )
                                    with col1_km:
                                        st.write("##")
                                        st.plotly_chart(fig, use_container_width=True)

                            elif len(st.session_state.classes_SVM) > 2:
                                with col1_km:
                                    st.warning("Saisie invalide - trop de colonne selectionnées")

                    else:
                        with col1_km:
                            st.warning("Le dataset ne contient qu'une classe")
        elif len(st.session_state.choix_col_SVM) > 2:
            with col1_km:
                st.warning("Saisie invalide - trop de colonne selectionnées")


    else:
        with exp2:
            st.write("##")
            st.info('Rendez-vous dans la section Dataset pour importer votre dataset')
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)

elif st.session_state.choix_page_classification == "🪛 Decision Tree":
    st.markdown('<p class="grand_titre">Decision Tree</p>', unsafe_allow_html=True)
    st.write("##")
    exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
    if 'data' in st.session_state:
        st.markdown('<p class="section">Selection des features et de la target</p>', unsafe_allow_html=True)
        _, col1_dt, _ = st.columns((0.1, 1, 0.1))
        _, col1_eval_modele, col2_eval_modele, col3_eval_modele, col4_eval_modele, _ = st.columns(
            (0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
        _, col_res, _ = st.columns((0.1, 1, 0.1))
        with col1_dt:
            st.session_state.choix_col_DT = st.multiselect("Choisir deux colonnes",
                                                            col_numeric(st.session_state.data),
                                                            help="Vos features", max_selections=2,
                                                            
                                                            )
            st.session_state.choix_target_DT = st.selectbox("Choisir la target",
                                                            st.session_state.data.columns.tolist()[::-1],
                                                            )
        if len(st.session_state.choix_col_DT) > 0:
            target = st.session_state.choix_target_DT
            features = st.session_state.choix_col_DT

            # dataset avec features + target
            df = st.session_state.data[[target] + features]
            df = df.dropna(axis=0)

            if len(df) == 0:
                with col1_dt:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                if st.session_state.choix_target_DT in st.session_state.choix_col_DT:
                    with col1_dt:
                        st.warning("La target ne doit pas appartenir aux features")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], random_state=0)
                    model = DecisionTreeClassifier(random_state=0)
                    model.fit(X_train, y_train)
                    if len(pd.unique(df[target])) > 2:
                        average = 'macro'
                    else:
                        average = 'binary'

                    # metrics on train
                    y_pred_train = model.predict(X_train)
                    accur_train = accuracy_score(y_train, y_pred_train)
                    precis_train = precision_score(y_train, y_pred_train, average=average,
                                                    pos_label=pd.unique(df[target])[0])
                    rappel_train = recall_score(y_train, y_pred_train, average=average,
                                                pos_label=pd.unique(df[target])[0])
                    F1_train = f1_score(y_train, y_pred_train, average=average, pos_label=pd.unique(df[target])[0])

                    # metrics on test
                    y_pred_test = model.predict(X_test)
                    accur_test = accuracy_score(y_test, y_pred_test)
                    precis_test = precision_score(y_test, y_pred_test, average=average,
                                                    pos_label=pd.unique(df[target])[0])
                    rappel_test = recall_score(y_test, y_pred_test, average=average,
                                                pos_label=pd.unique(df[target])[0])
                    F1_test = f1_score(y_test, y_pred_test, average=average, pos_label=pd.unique(df[target])[0])

                    # Affichage métriques
                    with col1_dt:
                        st.write("##")
                        st.markdown(
                            '<p class="section">Évaluation par rapport au train set</p>',
                            unsafe_allow_html=True)
                        st.write("##")
                    with col1_eval_modele:
                        st.metric(label="Precision", value=round(precis_test, 3),
                                    delta=round(precis_test - precis_train, 3))
                    with col2_eval_modele:
                        st.metric(label="Recall", value=round(rappel_test, 3),
                                    delta=round(rappel_test - rappel_train, 3))
                    with col3_eval_modele:
                        st.metric(label="F1 score", value=round(F1_test, 3),
                                    delta=round(F1_test - F1_train, 3))
                    with col4_eval_modele:
                        st.metric(label="Accuracy", value=round(accur_test, 3),
                                    delta=round(accur_test - accur_train, 3))
                    with col_res:
                        st.write("##")
                        st.markdown(
                            '<p class="section">Résultat arbre de décision</p>',
                            unsafe_allow_html=True)
                        st.write("##")
                    # DOT data
                    dot_data = export_graphviz(model, out_file=None,
                                                feature_names=features,
                                                class_names=target,
                                                filled=True,
                                                )
                    # Draw graph
                    st.graphviz_chart(dot_data, use_container_width=False)

    else:
        with exp2:
            st.write("##")
            st.info('Rendez-vous dans la section Dataset pour importer votre dataset')
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)

elif st.session_state.choix_page_classification == "🪛 Logistic regression":
    st.markdown('<p class="grand_titre">Logistic Regression</p>', unsafe_allow_html=True)
    st.write("##")
    exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
    if 'data' in st.session_state:
        st.markdown('<p class="section">Selection des features et de la target</p>', unsafe_allow_html=True)
        _, col1_lr, _ = st.columns((0.1, 1, 0.1))
        _, col1_eval_modele, col2_eval_modele, col3_eval_modele, col4_eval_modele, _ = st.columns(
            (0.3, 0.5, 0.5, 0.5, 0.5, 0.1))
        _, col_res, _ = st.columns((0.1, 1, 0.1))
        with col1_lr:
            st.session_state.choix_col_LR = st.multiselect("Selection des features",
                                                            col_numeric(st.session_state.data),
                                                            help="Vos features",
                                                            
                                                            )
            st.session_state.choix_target_LR = st.selectbox("Choisir la target",
                                                            st.session_state.data.columns.tolist()[::-1],
                                                            help="Choisir une variable catégorielle"
                                                            )
        if len(st.session_state.choix_col_LR) > 0:
            features = st.session_state.choix_col_LR
            target = st.session_state.choix_target_LR

            # dataset avec features + target
            df = st.session_state.data[[target] + features]
            df = df.dropna(axis=0)

            if len(df) == 0:
                with col1_lr:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                if target in features:
                    with col1_lr:
                        st.warning("La target ne doit pas appartenir aux features")
                else:
                    if len(df[target].unique().tolist()) > 2:
                        with col1_lr:
                            st.session_state.classes_LR = st.multiselect("Choisir deux classes",
                                                                            df[st.session_state.choix_target_LR].unique().tolist(), max_selections=2,
                                                                            )
                        
                        if len(st.session_state.classes_LR) == 2:
                            df = df.loc[
                                    (df[target] == st.session_state.classes_LR[0]) | (
                                            df[target] == st.session_state.classes_LR[1])]


                    if len(df[target].unique().tolist()) == 2:
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], random_state=0)
                            model = make_pipeline(StandardScaler(), LogisticRegression())
                            model.fit(X_train, y_train)

                            if len(pd.unique(df[target])) > 2:
                                average = 'macro'
                            else:
                                average = 'binary'
                            # metrics on train
                            y_pred_train = model.predict(X_train)
                            accur_train = accuracy_score(y_train, y_pred_train)
                            precis_train = precision_score(y_train, y_pred_train, average=average,
                                                            labels=pd.unique(df[target]), pos_label=pd.unique(df[target])[0])
                            rappel_train = recall_score(y_train, y_pred_train, average=average,
                                                        labels=pd.unique(df[target]), pos_label=pd.unique(df[target])[0])
                            F1_train = f1_score(y_train, y_pred_train, average=average, labels=pd.unique(df[target]), pos_label=pd.unique(df[target])[0])

                            # metrics on test
                            y_pred_test = model.predict(X_test)
                            accur_test = accuracy_score(y_test, y_pred_test)
                            precis_test = precision_score(y_test, y_pred_test, average=average,
                                                            labels=pd.unique(df[target]), pos_label=pd.unique(df[target])[0])
                            rappel_test = recall_score(y_test, y_pred_test, average=average,
                                                        labels=pd.unique(df[target]), pos_label=pd.unique(df[target])[0])
                            F1_test = f1_score(y_test, y_pred_test, average=average, labels=pd.unique(df[target]), pos_label=pd.unique(df[target])[0])

                            # Affichage métriques
                            with col1_lr:
                                st.write("##")
                                st.markdown(
                                    '<p class="section">Évaluation par rapport au train set</p>',
                                    unsafe_allow_html=True)
                                st.write("##")
                            with col1_eval_modele:
                                st.metric(label="Precision", value=round(precis_test, 3),
                                            delta=round(precis_test - precis_train, 3))
                            with col2_eval_modele:
                                st.metric(label="Recall", value=round(rappel_test, 3),
                                            delta=round(rappel_test - rappel_train, 3))
                            with col3_eval_modele:
                                st.metric(label="F1 score", value=round(F1_test, 3),
                                            delta=round(F1_test - F1_train, 3))
                            with col4_eval_modele:
                                st.metric(label="Accuracy", value=round(accur_test, 3),
                                            delta=round(accur_test - accur_train, 3))
                            with col_res:
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
                                )
                                st.write("##")
                                st.markdown(
                                    '<p class="section">Learning curves</p>',
                                    unsafe_allow_html=True)
                                st.plotly_chart(fig, use_container_width=True)
                                st.write("##")
                                st.caption(
                                "Il est possible que votre dataset soit trop petit pour effectuer la cross-validation dans de bonnes conditions")
                                st.write("##")
                                st.write('---')

                        except ValueError:
                            with col1_lr:
                                st.write("##")
                                st.warning("Un problème est survenu lors de l'entrainement du modèle. Veuillez à bien choisir une target discrète.")

    else:
        with exp2:
            st.write("##")
            st.info('Rendez-vous dans la section Dataset pour importer votre dataset')
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)

elif st.session_state.choix_page_classification == "🪛 DBSCAN":
    st.markdown('<p class="grand_titre">DBSCAN</p>', unsafe_allow_html=True)
    st.write("##")
    exp1, exp2, exp3 = st.columns((0.2, 1, 0.2))
    if 'data' in st.session_state:
        _, col1_features_choice, _ = st.columns((0.1, 1, 0.1))
        _, slide_1, slide_2, _ = st.columns((0.1, 0.5, 0.5, 0.1))
        _, res, _ = st.columns((0.1, 1, 0.1))
        with col1_features_choice:
            st.write("##")
            st.markdown('<p class="section">Selection des features pour le modèle</p>', unsafe_allow_html=True)
            st.session_state.choix_col_dbscan = st.multiselect("Choisir au moins deux colonnes",
                                                                col_numeric(st.session_state.data),
                                                                
                                                                )
        if len(st.session_state.choix_col_dbscan) > 1:
            df_ml = st.session_state.data[st.session_state.choix_col_dbscan]
            df_ml = df_ml.dropna(axis=0)
            if len(df_ml) == 0:
                with col1_features_choice:
                    st.write("##")
                    st.warning('Le dataset avec suppression des NaN suivant les lignes est vide!')
            else:
                with slide_1:
                    st.write("##")
                    st.session_state.epsilon_dbscan = st.slider('Rayon de voisinage des points', min_value=0.1,
                                                                    max_value=5.1, step=0.2, value=2.1
                                                                    )
                with slide_2:
                    st.write("##")
                    st.session_state.min_sample_dbscan = st.slider('Nombre de voisins pour rejoindre un cluster', min_value=1,
                                                                max_value=10, step=1, value=4
                                                                )
                with res:
                    st.write("##")
                    X = df_ml[st.session_state.choix_col_dbscan]  # features
                    # try:
                    # DBSCAN
                    modele = DBSCAN(eps=st.session_state.epsilon_dbscan, min_samples=st.session_state.min_sample_dbscan)
                    labels_predicted_dbscan = modele.fit_predict(X)
                    X['labels'] = pd.Series(labels_predicted_dbscan)
                    # PCA
                    X_embedded = TSNE(n_components=2).fit_transform(df_ml[st.session_state.choix_col_dbscan])
                    X["x_component"] = X_embedded[:,0]
                    X["y_component"] = X_embedded[:,1]
                    fig = px.scatter(X, x="x_component", y="y_component", color='labels', color_discrete_sequence=px.colors.qualitative.G10)
                    fig.update(layout_coloraxis_showscale=False)
                    fig.update_layout(
                        showlegend=True,
                        template='simple_white',
                        font=dict(size=10),
                        autosize=False,
                        width=1000, height=650,
                        margin=dict(l=40, r=50, b=40, t=40),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.write("##")
                    st.markdown(
                        '<p class="section">Résultat après PCA</p>',
                        unsafe_allow_html=True)
                    st.write("##")
                    st.plotly_chart(fig, use_container_width=True)
                    #except:
                        #with col1_features_choice:
                            #st.write("##")
                            #st.error("Erreur de chargement")
    else:
        with exp2:
            st.write("##")
            st.info('Rendez-vous dans la section Dataset pour importer votre dataset')
            st.write("##")
            st_lottie(load_lottieurl('https://assets5.lottiefiles.com/packages/lf20_inuxiflu.json'), height=200)