<p align="center">
  <img src="https://user-images.githubusercontent.com/63207451/141209252-a98cc392-8831-4fbe-af90-61cb7eee8264.png" height="100">
  <img src="https://user-images.githubusercontent.com/63207451/141208795-3b0b5e6e-e014-4215-8ed2-fdd205ddfa41.png" height="100">
  <img src="https://user-images.githubusercontent.com/63207451/145711302-b7184614-9c46-43b1-9448-0640ecfdc6de.png" height="100">
  <img src="https://user-images.githubusercontent.com/63207451/118670736-29bd2980-b7f7-11eb-8aa4-ad41fa393ed1.png" height="100">
</p>

<br/>
<h1 align="center"> No-code AI platform </h1>
<br/>

<p align="center">
  <a href="https://antonin-lfv-online-preprocessing-for-ml-accueil-dh2xej.streamlit.app"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a>
  </p>

<br/>

<p align="center">
  La <b>No-code AI platform</b> est un site développé avec <b>Python</b> et déployé avec <b>Streamlit</b>, qui permet de faire du Machine/Deep Learning sans écrire de code. La partie analyse et Machine Learning repose sur l'utilisation d'un dataset, qui peut être soit un dataset déjà disponible sur le site (les iris, ou les penguins), soit un dataset de votre choix que vous aurez uploadé, et avec qui vous pourrez effectuer du preprocessing directement depuis la page d'upload (Attention à bien re télécharger le dataset modifié et de le re uploader). Une fois le dataset choisi, vous pouvez l'utiliser pour alimenter des algorithmes tels que des SVM, des K-Means, des KNN ou encore des réductions de dimension.
<p/>
<br>

<p align="center">
<b>La No-Code AI Platform a été ensuite créée avec Flask et déployée sur Heroku, le repo est disponible <a href="https://github.com/antonin-lfv/No-code-AI-platform">ici</a></b>
  </p>
  
  <br>  <br>  <br>  <br>
  
> Pour accéder au site, cliquez sur le bouton **"Open in streamlit"** en haut du ReadMe

<br>

## Test du site en local

Clonez le repo puis, depuis le terminal au dossier du projet :

```bash
Streamlit run Accueil.py
```

<br/>

## Mode d'emploi

Je vais détailler ici l'utilisation de la platform, avec des exemples pour chacune des pages. Pour commencer, voici la première par laquelle vous allez arriver. Sur la gauche se trouve la barre de navigation, vous permettant de vous déplacer au travers des pages. À partir de cette page, il vous faut charger des données sur la page **Dataset**, sous peine de ne pas pouvoir utiliser les autres pages.

<p align="center">
  <img width="850" alt="Capture d’écran 2022-10-27 à 23 28 53" src="https://user-images.githubusercontent.com/63207451/198401805-e4a95b7e-51d3-4579-ac6b-27d37d76494c.png">
  </p>

Détaillons maintenant la page Dataset. Sur cette page, vous allez choisir les données à utiliser. Pour ce faire, plusieurs choix s'offrent à vous. Le premier est d'utiliser l'un des dataset proposés dont le type de problème (Classification ou Régression) est donné. Le second est d'utiliser votre propre dataset que vous importerai en cliquant sur **Choisir un dataset personnel**.

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-27 à 23 32 16" src="https://user-images.githubusercontent.com/63207451/198402345-a71ecd6a-debc-4e21-9f00-b1d0169b30d4.png">
  </p>


Si vous choisissez un dataset personnel, vous pourrez spécifier un séparateur si besoin, et modifier le dataset (il faudra retélécharger le dataset après modification). Dans le cas ou vous choisissez un dataset parmi ceux proposés, vous ne pourrez pas les modifier (ils sont déjà nettoyés).

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-27 à 23 32 16" src="https://user-images.githubusercontent.com/63207451/198404597-41d4b83e-369c-4528-b511-a59047e31ec1.png">
  </p>
  
<p align="center">
<img width="850" alt="Capture d’écran 2022-10-27 à 23 32 16" src="https://user-images.githubusercontent.com/63207451/198404609-a4e73f9d-53a1-4c05-9ca4-5d4480a77cb7.png">
  </p>

À présent, vous pouvez naviguer parmi les pages pour visualiser les données, créer des modèles, etc. La première page est celle d'**analyse des colonnes**, elle permet d'afficher les colonnes indépendemment, et d'avoir les caractéristiques mathématiques de celles-ci. Les caractéristiques dépendent du type de la colonne.

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-27 à 23 32 16" src="https://user-images.githubusercontent.com/63207451/198405223-96e1acb5-c875-40a0-84bc-5733b4b79319.png">
  </p>


La page suivant, **Matrice de corrélations** permet de voir les possibles corrélations entre les features du dataset. Vous devez simplement spécifier les features à prendre en compte. Vous pouvez également renseigner une feature catégorielle pour la coloration.

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-27 à 23 32 16" src="https://user-images.githubusercontent.com/63207451/198405775-12b0cf2d-a26c-4226-9ca5-8f9ae2fcde2a.png">
  </p>

La page **section graphiques** vous donne la possibilité de créer des graphiques pour suivre en détail les corrélations entre deux features. Plusieurs options de graphiques sont disponibles (Points, Courbe, Coordonnées géographiques, Histogramme), plusieurs indices statistiques peuvent être affichés ainsi que les courbes de régressions Linéaire et Polynomiale avec les équations finales pour chacune.

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-27 à 23 32 16" src="https://user-images.githubusercontent.com/63207451/198406497-c46770c5-e80c-4e04-aaf7-cf4db1707b0a.png">
  </p>

Nous voila maintenant dans la section **Régressions**, dans cette page vous pourrez suivre les courbes d'apprentissage de plusieurs modèles de régressions (sélectionnés en fonction du type de données) ainsi que l'évolution des métriques. Il y a la régression linéaire, polynomiale, Elastic Net, Ridge, Lasso, de poisson. Il vous suffit de renseigner les features et la bonne target et le tour est joué.

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-27 à 23 32 16" src="https://user-images.githubusercontent.com/63207451/198407133-6888d8a1-0f27-4322-9ca1-71229e4abe7c.png">
  </p>

Attaquons nous maintenant à la partie classification, qui renferme des nouvelles pages pour chaque modèle, à savoir le KNN, K-Means, les Support Vector Machines et les Decision Trees. 
Le premier modèle est le modèle des k plus proches voisins (KNN). Pour créer le modèle, il faudra commencer par sélectionner au moins deux colonnes, dont la target du modèle. Vous spécifierai les données à encoder, et vous choisierai la target (parmi les colonnes sélectionnées précedemment) qui doit être encodée. L'encodage peut être visualisé dans une petite fenetre. 
Après avoir lancé le modèle, vous verrez le nombre optimal de voisins pour avoir les meilleures performances ainsi que les courbes d'apprentissages, les métriques et les courbes ROC dans le cas d'une classification binaire.

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 19 55 37" src="https://user-images.githubusercontent.com/63207451/198701520-eb8663b5-9a92-46ec-a59f-f9d5a5464c2a.png">
  </p>
  
<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 19 57 28" src="https://user-images.githubusercontent.com/63207451/198701827-876d9de1-2d8a-4726-b09b-415d29a21996.png">
  </p>
  
<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 19 59 09" src="https://user-images.githubusercontent.com/63207451/198702087-b9e7b51f-8df8-4c2d-ad3d-4e7148f88d0d.png">
  </p>

Il vous sera aussi possible d'utiliser le modèle pour faire des prédictions.

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 20 01 35" src="https://user-images.githubusercontent.com/63207451/198702474-bf70eeed-3e74-4ec6-a49c-65b9dac45819.png">
  </p>

Poursuivons sur le prochain algorithme, qui est le K-means. Pour créer le modèle il faudra renseigner au moins deux colonnes, et modifier ou non le nombre de clusters que le modèle doit créer. Le résultat sera affiché grâce à l'algorithme PCA, avec coloration des clusters et des centroides.

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 20 04 52" src="https://user-images.githubusercontent.com/63207451/198703040-256abf65-d646-434e-89bc-057b0126694a.png">
  </p>
  
<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 20 05 01" src="https://user-images.githubusercontent.com/63207451/198703062-90e87dc4-7676-4840-b989-bfd077d404ab.png">
  </p>

Il est également possible de créer un modèle de support vector machine (SVM). Pour cela, on choisiera deux colonnes pour les features, la target qui doit être catégorielle et dont on sélectionnera uniquement deux classes, et le noyau qui est pour l'instant uniquement linéaire. Une fois le modèle lancé vous pourrez visualiser les résultats. 

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 20 10 39" src="https://user-images.githubusercontent.com/63207451/198704113-83303801-1937-4212-a216-a8be2c1f37d0.png">
  </p>
  
<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 20 10 46" src="https://user-images.githubusercontent.com/63207451/198704136-a2458246-a1c0-449e-ae6c-3642cced249d.png">
  </p>

Enfin, le dernier algorithme de classification est celui des arbres de décision. Comme précedemment il vous suffit de sélectionner deux features ainsi que la target pour créer et visualiser le modèle.

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 20 23 18" src="https://user-images.githubusercontent.com/63207451/198706091-c6639432-d52d-4a81-8417-8ca942aaadd5.png">
  </p>

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 20 23 56" src="https://user-images.githubusercontent.com/63207451/198706202-e17a4315-7ed7-49f0-b5fe-6bfa5b05e90d.png">
  </p>

Enfin, voici la dernière section de la No code AI platform, concernant la réduction de dimension. Le premier algorithme est l'analyse en composantes principales. Il faudra sélectionner au moins deux colonnes des données, dont la target, ainsi que les colonnes à encoder et la target (qui appartient aux colonnes séléctionnées au début).

<p align="center">
<img width="850" alt="Capture d’écran 2022-10-28 à 20 31 15" src="https://user-images.githubusercontent.com/63207451/198707454-2aa55307-7d44-4d03-a5cf-85bafe2f6d99.png">
  </p>
  
<p align="center">  
<img width="850" alt="Capture d’écran 2022-10-28 à 20 31 34" src="https://user-images.githubusercontent.com/63207451/198707494-9958e5cc-d30b-41e7-8c95-c090a0106057.png">
  </p>

Le deuxième algorithme de réduction de dimension est l'algorithme UMAP, qui marche de la même manière que le précédent.

<p align="center">  
<img width="850" alt="Capture d’écran 2022-10-28 à 20 34 02" src="https://user-images.githubusercontent.com/63207451/198707906-14879f08-1cf1-47b0-bbab-0e8975e5aa2a.png">
  </p>

<p align="center">  
<img width="850" alt="Capture d’écran 2022-10-28 à 20 34 58" src="https://user-images.githubusercontent.com/63207451/198708061-37dfcb80-9ce9-43e2-96f2-4809088d8c95.png">
  </p>

Enfin, vous pouvez utiliser l'algorithme t-SNE, qui s'utilise comme les deux algorithmes précédents.

<p align="center">  
<img width="850" alt="Capture d’écran 2022-10-28 à 20 40 26" src="https://user-images.githubusercontent.com/63207451/198708968-5e3b356c-20e3-41fa-ad46-e1b011a36f0e.png">
  </p>
  
<p align="center">  
<img width="850" alt="Capture d’écran 2022-10-28 à 20 40 33" src="https://user-images.githubusercontent.com/63207451/198708991-3ffad937-7486-4d43-8596-19b474c59007.png">
  </p>
  
<br>

<p align="center">
    <a href="https://antonin-lfv.github.io" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/127334786-f48498e4-7aa1-4fbd-b7b4-cd78b43972b8.png" title="Web Page" width="38" height="38"></a>
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>
