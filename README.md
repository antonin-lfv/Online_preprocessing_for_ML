<p align="center">
  <img src="https://user-images.githubusercontent.com/63207451/122607703-45785180-d07b-11eb-9f7e-b8505e04d5b1.png" height="100">
</p>
  
<h1 align="center">Online preprocessing & Machine Learnning</h1>
<br/>

<p align="center">
  <a href="https://share.streamlit.io/antonin-lfv/online_preprocessing_for_ml/main.py"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a>
  </p>

<br/>

<p align="center">
Cette application web developpée en python et publiée grâce à Streamlit a pour but d'accélérer l'exploration des données, et la mise en place d'algorithmes de machine learning. Elle est basée sur un système multi-pages, qui conserve les widgets entre celles-ci grâce à un système de Session. Le principe est simple, vous déposez votre dataset sous le format csv ou xls sur le volet gauche, puis le Menu apparaîtra. À partir de là il faudra se rendre dans la section <b>Chargement du dataset</b> pour l'importer, et effectuer des réglages sur celui-ci comme ajouter un séparateur, transformer des colonnes en Time-Series, ou supprimer des symboles qui empêchent un typage en float. <br/>
Puis, libre à vous de faire votre analyse des données, observer les colonnes, leurs statistiques, les corrélations entre elles. Depuis la section <b>graphiques et regressions</b> vous pourrez tracer des graphiques avec des points, des courbes, des histogrammes ou même des coordonnées géographiques et effectuer des regressions linéaires et polynomiales. <br/>
Enfin, vous avez la possibilité de créer des modèles de machine learning rapidement comme des KNN pour effectuer des prédictions.
  </p>
  
<br/>

# Interface

## Home

En arrivant sur le DashBoard, vous verrez ceci :
<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/63207451/122610535-1dd7b800-d080-11eb-8416-c30390474f36.png" height="600">
</p>

<br/>

<br/>
Il vous faut ici choisir un dataset csv ou excel de votre machine, en cliquant sur le bouton <b>Browse files</b>. Une fois cela effectuée, le menu lateral apparaîtra. Pour ensuite pouvoir utiliser votre fichier, il faut vous rendre à la page 2, nommée <b>Chargement du dataset</b>. <br/>
Sur cette page, plusieurs modifications sont possibles sur le dataset : 
- Ajout d'un séparateur si besoin
- Conversion de colonnes en Time-Series
- Conversion de colonnes contenant des symboles monétaires en float
- Conversion de colonnes de strings de nombres à virgules en float
De plus, les caractéristiques principales de ce dataset sont affichées sur le coté.<br/>

<br/>
<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/63207451/122611487-b1f64f00-d081-11eb-9af5-2fbb85e9e3bf.png" height="600">
</p>

<br/>

<br/>

<br/>


<p align="center">
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>
