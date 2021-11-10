<p align="center">
  <img src="https://user-images.githubusercontent.com/63207451/122607703-45785180-d07b-11eb-9f7e-b8505e04d5b1.png" height="100">
  <img src="https://user-images.githubusercontent.com/63207451/141207902-87510f35-c5f9-482a-8194-80b782a17f49.png" height="100">
</p>

<br/>
<h1 align="center"> No-code AI platform </h1>
<br/>

<p align="center">
  <a href="https://share.streamlit.io/antonin-lfv/online_preprocessing_for_ml/main.py"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a>
  [![Codacy Badge](https://app.codacy.com/project/badge/Grade/24ff1896d8a548b2a6800b6836bf21fb)](https://www.codacy.com/gh/antonin-lfv/Online_preprocessing_for_ML/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=antonin-lfv/Online_preprocessing_for_ML&amp;utm_campaign=Badge_Grade)
  </p>

<br/>

<p align="center">
Cette application web developpée en python et publiée grâce à Streamlit a pour but d'accélérer l'exploration des données, et la mise en place d'algorithmes de machine learning. Elle est basée sur un système multi-pages, qui conserve les widgets entre celles-ci grâce à un système de Session. Le principe est simple, vous déposez votre dataset sous le format csv ou xls sur le volet gauche, puis à partir de là, il faudra se rendre dans la section <b>Dataset</b> pour l'importer, et effectuer des réglages sur celui-ci comme ajouter un séparateur, transformer des colonnes en Time-Series, ou supprimer des symboles qui empêchent un typage en float. <br/>
Puis, libre à vous de faire votre analyse des données, observer les colonnes, leurs statistiques, les corrélations entre elles. Depuis la <b>section graphiques </b> vous pourrez tracer des graphiques avec des points, des courbes, des histogrammes ou même des coordonnées géographiques et effectuer des regressions linéaires et polynomiales. <br/>
Enfin, vous avez la possibilité de créer des modèles de machine learning ou de deep learning rapidement.  <br/>
Dans la section ML, vous sera possible d'effectuer des KNN, K-Means, SVM, PCA et UMAP. Quant à la section DL, vous avez la possibilité de faire du transfert de style neuronal, avec plusieurs images déjà disponibles. <br/>
En cas de bug important, veuillez me le signaler pour qu'il puisse être corrigé le plus rapidement possible.
  </p>
  
<br/>

# Index

1. [Interface](#Interface)
    - [Accueil](#Accueil)
    - [Chargement du dataset](#Chargement-du-dataset)
    - [Analyse des colonnes](#Analyse-des-colonnes)
    - [Matrice de corrélation](#Matrice-de-corrélation)
    - [Section graphiques](#Section-graphiques)
    - [Machine Learning](#Machine-Learning)
      - [k-nearest neighbors](#k-nearest-neighbors)
      - [k-Means](#K-Means)
      - [Support Vector Machine](#Support-Vector-Machine)
      - [PCA](#PCA)
      - [UMAP](#UMAP)
    - [Deep Learning](#Deep-Learning)
      - [Transfert de style neuronal](#Transfert-de-style-neuronal)
      - [GAN](#GAN)


<br/>

# Interface

## Accueil

En arrivant sur le DashBoard, vous verrez ceci :
<br/>

<p align="center">
  <img width="1439" alt="Capture d’écran 2021-07-22 à 18 56 51" src="https://user-images.githubusercontent.com/63207451/126678415-b7980d3d-1364-45d2-9b88-4b73aae43ca6.png">
</p>

<br/>

<br/>
Il vous faut ici choisir un dataset csv ou excel de votre machine, en cliquant sur le bouton <b>Browse files</b>. Une fois cela effectuée, le menu lateral apparaîtra. Pour ensuite pouvoir utiliser votre fichier, il faut vous rendre à la page 2, nommée <b>Chargement du dataset</b>. <br/>

## Chargement du dataset

Sur cette page, plusieurs modifications sont possibles sur le dataset : 
- Ajout d'un séparateur si besoin
- Conversion de colonnes en Time-Series
- Conversion de colonnes contenant des symboles monétaires en float
- Conversion de colonnes de strings de nombres à virgules en float <br/>

De plus, les caractéristiques principales de ce dataset sont affichées sur le coté.<br/>

<br/>
<br/>

<p align="center">
  <img width="1439" alt="Capture d’écran 2021-07-22 à 18 57 47" src="https://user-images.githubusercontent.com/63207451/126678557-f687704b-535d-47bf-801b-3008c926e1ef.png">
</p>

<br/>

## Analyse des colonnes

<br/>
<p align="center">
  <img width="1439" alt="Capture d’écran 2021-07-22 à 18 58 24" src="https://user-images.githubusercontent.com/63207451/126678641-70c74b1d-0bde-4215-91ce-28ec5d9e5ac7.png">
</p>

<br/>

## Matrice de corrélation

<br/>
<p align="center">
  <img width="1439" alt="Capture d’écran 2021-07-22 à 18 59 36" src="https://user-images.githubusercontent.com/63207451/126678828-a2d20126-694c-4c9b-9613-ee1d6a3ef466.png">
</p>

<br/>


## Section graphiques

<br/>
<p align="center">
<img width="1439" alt="Capture d’écran 2021-07-22 à 19 02 28" src="https://user-images.githubusercontent.com/63207451/126679232-d2968749-9b57-40ba-9543-5410342abd3c.png">
</p>

<br/>

## Machine Learning

<br/>

### k-nearest neighbors

<br/>
<p align="center">
<img width="1439" alt="Capture d’écran 2021-07-22 à 19 13 06" src="https://user-images.githubusercontent.com/63207451/126680576-11993544-5da2-4755-b235-ed513c0aa9a3.png">
</p>

<br/>

### k-Means

<br/>
<p align="center">
<img width="1439" alt="Capture d’écran 2021-07-22 à 19 13 58" src="https://user-images.githubusercontent.com/63207451/126680672-2e178b35-d8fc-49e5-b179-a4fdd30402c0.png">
</p>

<br/>

### Support Vector Machine

<br/>
<p align="center">
<img width="1439" alt="Capture d’écran 2021-07-22 à 19 15 01" src="https://user-images.githubusercontent.com/63207451/126680798-816687fd-eac9-4d4f-a817-25f9ee9876d1.png">
</p>

<br/>

### PCA

<br/>
<p align="center">
<img width="1439" alt="Capture d’écran 2021-07-22 à 19 16 15" src="https://user-images.githubusercontent.com/63207451/126680981-247d81d6-49f4-45e9-8c40-01526a237e58.png">
</p>

<br/>

### UMAP

<br/>
<p align="center">
<img width="1439" alt="Capture d’écran 2021-07-22 à 19 16 40" src="https://user-images.githubusercontent.com/63207451/126681039-aa71207a-96ae-4110-80e2-9ab8a4e59b2e.png">
</p>

<br/>

## Deep Learning

<br/>

### Transfert de style neuronal

<br/>
<p align="center">
<img width="1439" alt="Capture d’écran 2021-07-22 à 19 17 12" src="https://user-images.githubusercontent.com/63207451/126681123-f05da0b5-e0d8-496c-8c92-ef8ae10caccb.png">
</p>

<br/>

### GAN

<br/>

Cette section est en développement.

<p align="center">
<img width="1439" alt="Capture d’écran 2021-07-22 à 19 17 34" src="https://user-images.githubusercontent.com/63207451/126681154-fa50acbe-766b-44a1-89a6-43288f06d6f6.png">
</p>

<br/>

<br/>

<p align="center">
    <a href="https://antonin-lfv.github.io" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/127334786-f48498e4-7aa1-4fbd-b7b4-cd78b43972b8.png" title="Web Page" width="38" height="38"></a>
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>
