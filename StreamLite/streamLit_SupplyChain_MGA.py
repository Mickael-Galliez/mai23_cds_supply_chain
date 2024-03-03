#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:07:36 2024

@author: Asselot Joan && Mickael Galliez
"""
# Importation des librairies.
import streamlit as st
import joblib
import pandas as pd
import nltk
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack

# Téléchargement de la liste de stop words en français.
nltk.download('stopwords')

# Importation de la liste de stop words en français.
from nltk.corpus import stopwords
stop_words_french = set(stopwords.words('french'))

df = pd.read_csv("df1.csv")
df2 = pd.read_csv("df2.csv")

def get_top_ngram(corpus, n=None, ngrame=1,stopWord=True):
    """
    Compte le nombre de recurrences des ngrames presents dans le corpus et retourne le top n ngrames et leur decomptes.
    :param corpus: Serie, incluant 1 seul colonne de string
    :param n: int, Nombre de Ngrame a collecter apres triage
    :param ngrame: int, ordre du Ngrame vise. 1 mot = 1grame, couple mots se suivant = 2grame, etc
    :param stopWord: Boolean, filtre les stopWord present dans les string. La list de stop word est definie dans la
    fonction. StopWord example = le,la,les,de,je,etc
    :return: List of tuple, [(mot,nombre recense)]
    """
    items = {"ä": "a", "ç": "c", "è": "e", "º": "", "Ã": "A", "Í": "I", "í": "i", "Ü": "U", "â": "a", "ò": "o", "¿": "",
             "ó": "o", "á": "a", "à": "a", "õ": "o", "¡": "", "Ó": "O", "ù": "u", "Ú": "U", "´": "", "Ñ": "N", "Ò": "O",
             "ï": "i", "Ï": "I", "Ç": "C", "À": "A", "É": "E", "ë": "e", "Á": "A", "ã": "a", "Ö": "O", "ú": "u",
             "ñ": "n", "é": "e", "ê": "e", "·": "-", "ª": "a", "°": "", "ü": "u", "ô": "o","+":"plus","-":"moins","_":" "}
    stopWordFrench = ['alors','au','ai','aucuns','aussi','autre','avant','avec','avoir','bon','car','ce','cela',
                      'ces','ceux','chaque','ci','comme','comment','dans','de','des','du','dedans','dehors','depuis',
                      'devrait','doit','donc','dos','debut','elle','elles','en','encore','essai','est','et','eu',
                      'fait','faites','fois','font','hors','ici','il','ils','je','juste','la','le','les','leur','ma',
                      'maintenant','mais','mes','mien','moins','mon','mot','meme','ni','nommes','notre','nous','ou',
                      'par','parce','peut','plupart','pour','pourquoi','quand','que','quel','quelle','quelles',
                      'quels','qui','sa','sans','ses','seulement','si','sien','son','sont','sous','soyez','sujet',
                      'sur','ta','un','une','tandis','tellement','tels','tes','ton','tous','tout','tres','tu',
                      'voient','vont','votre','vous','vu','ca','etaient','etat','etions','ete','etre','me','chez',
                      'on','ont',"de_","et_","la_","le_","j_ai","j_"]
    stopWord = stopWordFrench if stopWord else None
    corpus = corpus.str.replace(r'[^\x00-\x7F]', lambda x: items.get(x.group(0)) or '_', regex=True)
    vec = CountVectorizer(ngram_range=(ngrame, ngrame), stop_words=stopWord).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def Count_special(str):
    upper, lower, number, ponctuation, special = 0, 0, 0, 0, 0
    for i in range(len(str)):
        if str[i].isupper():
            upper += 1
        elif str[i].islower():
            lower += 1
        elif str[i].isdigit():
            number += 1
        elif str[i] in ['!','?']:
            ponctuation += 1
        elif str[i] in '@#$%&+=-<>~/\"*(){}[]':
            special += 1  
    return upper,lower,number,ponctuation,special
print(Count_special(df.Titre[0]))
df['UpperCount'], _, df['digitCount'], df['PonctCount'],df['SpecialCount'] = zip(*df.Commentaire.apply(lambda x: Count_special(x)))

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gensim.models import FastText
from sklearn.metrics import RocCurveDisplay, precision_recall_curve, auc, PrecisionRecallDisplay
import shap

from keras.models import load_model

from streamlit_extras.let_it_rain import rain

# Téléchargez la liste de stop words en français
nltk.download('stopwords')
nltk.download('punkt')

# Importez la liste de stop words en français
from nltk.corpus import stopwords

@st.cache_resource
def get_df(fileName):
    """
    Renvoie le fichier csv designe par fileName sous forme de dataframe
    :param fileName: string
    :return: dataframe
    """
    return pd.read_csv(fileName)
@st.cache_resource
def get_fulldfFastText():
    """
    Retourne un dataframe avec les colonnes commentaires et titres vectorises.
    Cette fonction est utile pour les sections modelisation II et III.
    On appelle la fonction en debut de fonction principale pour charger les donnees dans le cache
    :return:
    """

    df = get_df("df_featEngin.csv").loc[0:10000]
    df['embedding_Commentaire'] = df['Commentaire'].apply(get_text_embedding)
    df['embedding_Titres'] = df['Titres'].apply(get_text_embedding)
    df_expanded = df['embedding_Commentaire'].apply(lambda x: pd.Series(x, dtype="float64")).add_prefix(
        'vectComment_')
    df_expanded2 = df['embedding_Titres'].apply(lambda x: pd.Series(x, dtype="float64")).add_prefix('vectTitre_')
    df = pd.concat([df_expanded,df_expanded2,df[['Nombre_avis_publie','longCommentaire','longTitres','nb_Mots_Commentaire',
        'nb_Mots_Titres', 'nb_majuscules_Commentaire',
           'nb_chiffres_Commentaire', 'nb_ponctuation_Commentaire',
           'nb_special_Commentaire', 'nb_majuscules_Titre', 'nb_chiffres_Titre',
           'nb_ponctuation_Titre', 'nb_special_Titre', 'Sentiment']]],axis=1)

    return df


@st.cache_resource
def get_fastTextModel():
    """
    renvoie le model fast text. Ce model est long a construire aussi nous utiliserons le decorateur cache ressource
    cette fonction doit etre appele une premiere fois au debut de la fonction principale afin de charge le model dans le cache
    elle pourra etre appele a nouveau par la suite dans les differentes sections sans temps de chargement.
    :return: model fast text
    """
    # Replace 'path/to/pretrained/embeddings' with the actual path to your downloaded embeddings file
    return FastText.load_fasttext_format("FastText/cc.fr.300.bin")

def get_splited_df(data,target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

def preprocess_text(text):
    """
    Applique un nettoyage puis tokenize le texte. Etape necessaire avant la vectorization
    :param text: String
    :return: [string] list of string
    """
    stop_words = set(stopwords.words('french'))

    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return tokens

def get_text_embedding(text):
    """
    Vectorize le text a partir du model preentrainne Fast-text
    :param text: [String] List of String
    :return: [float] Vector list of float
    """
    model = get_fastTextModel()

    preprocessed_text = preprocess_text(text)
    word_embeddings = [model.wv[word] for word in preprocessed_text if word in model.wv.key_to_index]

    if word_embeddings:
        text_embedding = np.mean(word_embeddings, axis=0)  # Use mean for aggregation
    else:
        text_embedding = np.zeros(model.vector_size)  # Use zero vector if no embeddings found

    return text_embedding

@st.cache_resource
def get_saved_model(fileName):
    """
    Renvoie le model de classification .pkl designe par le parametre fileName
    :param fileName: String path du model
    :return: Model de classification
    """
    return joblib.load(fileName)

def classification_plots(y_test,y_pred):
    """
    Ensemble des graphs pertinents a notre probleme de classification
    :param y_test: list of int
    :param y_pred: list of int
    :return: None
    """
    colors = ["navy", "turquoise", "darkorange", "cornflowerblue", "teal"]



    # ROC curve
    plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_pred[:,1])
    plt.xlabel('False Positive Rate (Sentiment Positif)')
    plt.ylabel('True Positive Rate (Sensibilite)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt.gcf())


    # Precision-Recall:
    precision, recall, _ = precision_recall_curve(y_test, y_pred[:,0])
    # average_precision[i] = average_precision_score(y_t, y_p)

    plt.figure()
    PrecisionRecallDisplay.from_predictions(y_test,y_pred[:,0])
    # display = PrecisionRecallDisplay(
    # recall=recall,
    # precision=precision,
    # )
    # display.plot(name=f"Precision-recall Sentiment Negatif", color='navy')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt.gcf())

    return None

def rainningEmoji(emoji):
    rain(
        emoji=emoji,
        font_size=90,
        falling_speed=3,
        animation_length=30,
    )

def Count_special(str):
    upper, lower, number, ponctuation, special = 0, 0, 0, 0, 0
    for i in range(len(str)):
        if str[i].isupper():
            upper += 1
        elif str[i].islower():
            lower += 1
        elif str[i].isdigit():
            number += 1
        elif str[i] in ['!', '?']:
            ponctuation += 1

        elif str[i] in '@#$%&+=-<>~/\"*(){}[]':
            special += 1
    #     print('Upper case letters:', upper)
    #     print('Lower case letters:', lower)
    #     print('Number:', number)
    #     print('Ponctuation:', ponctuation)
    #     print('Special characters:', special)

    return upper, lower, number, ponctuation, special


#############################################################################################################
#                           Fonctions de Sections
#############################################################################################################
def demo_intro():
    st.image("wordCloud.png")
    st.write("#### Présentation générale")
    st.write("La première étape du projet consiste à collecter les données depuis le site Trustpilot. Dans un premier temps, nous avons sélectionné les 3 plus importantes sociétés référencées sur Trustpilot en nombres d’avis publiés et vérifiés: CentralePneus.fr, Allopneus et Pneus Online.")
    st.write("Dans un deuxième temps, nous nous sommes appuyés sur la technique du web scraping pour collecter l’ensemble des données disponibles et relatives à chacun des avis. Le web scraping est une technique permettant d’automatiser des processus de collecte de données sur le web à l’aide de robots ou de scripts automatisés appelés web crawlers.")
    st.write("Concernant la librairie, nous avons choisi BeautifulSoup plutôt que Selenium en raison de sa simplicité d’utilisation. BeautifulSoup fournit des méthodes simples pour naviguer, rechercher et modifier un arbre d’analyse dans des fichiers HTML ou XML. Cet outil permet non seulement de scraper, mais aussi de nettoyer les données. BeautifulSoup prend en charge l’analyseur HTML inclus dans la bibliothèque standard de Python, mais aussi plusieurs analyseurs Python tiers comme lxml ou html5lib.")
    st.write("Un avis client se présente de la manière suivante :")
    st.image("comment.png")
    st.write("   1.	**Client**, nom de l’utilisateur auteur de l'avis.")
    st.write("   2.	**Nombre_avis_publie**, nombre d’avis soumis par l’utilisateur.")
    st.write("   3.	**Pays**, pays d'origine de l’utilisateur.")
    st.write("   4.	**Note**, sur 5 attribuées par l’utilisateur.")
    st.write("   5.	**Statut**, de l’utilisateur. Le statut “vérifié” signifie que l’avis de cet utilisateur a été sollicité par le site Trustpilot à la suite de l'achat d’un service. Nous pouvons donc être très confiant sur l'honnêteté de cet avis. Si le symbole “avis vérifié” est absent, cela signifie que l’avis n’a pas fait l’objet d’une sollicitation et est donc moins fiable.")
    st.write("   6.	**Date_publication**, date de publication de l’avis.")
    st.write("   7.	**Titre**, de l’avis rédigé par l’utilisateur.")
    st.write("   8.	**Commentaire**, commentaire de l’avis rédigé par l’utilisateur.")
    st.write("   9. 	**Date_experience**, date de l'expérience utilisateur.")
    st.write("Ces données correspondent aux colonnes de notre jeu de données auxquels il faut ajouter les données suivantes :")
    st.write("  10.	**Entreprise**, le nom de la société concernée (n’apparaissant pas sur l’avis mais sur la page de l'avis).")
    st.write("  11.	**Date_reponse**, date de la réponse de la société lorsque celle-ci existe.")
    st.write("  12.	**Repons**, le commentaire de la réponse de la société lorsque celle-ci existe.")
    st.write("Une fois l'opération de web scraping terminée, nous avons obtenu un jeu de données composé de 12 colonnes et 286 000 entrées.")
    st.write("#### Variable explicative")
    st.write("À la suite d’une première analyse des données collectées, la variable explicative qui semble la plus appropriée pour déterminer la note attribuée par un utilisateur est Commentaire, c’est-à-dire le titre et le commentaire de l’avis rédigé par l’utilisateur.")
    st.write("#### Variable cible")
    st.write("L’objectif de ce projet étant de prédire la note attribuée en fonction des variables explicatives ci-dessus, la variable cible adéquate est Note. Dans le jeu de données initial, elle prend cinq valeurs différentes : ['1',  '2',  '3',  '4',  '5'].")
    st.write("#### Jeu de données:")
    df_visu = pd.read_csv("df1.csv")  
    st.write(df_visu.head(20))

def demo_visu():
  st.write("### Visualisation des données")
  st.write("##### Répartition des avis collectés par entreprise")
  avis_par_entreprise = df.groupby("Entreprise")["Titre"].count().reset_index()
  fig = plt.figure(figsize=(8, 6))
  sns.set_palette("pastel")  
  sns.set(font_scale=1.2)
  plt.pie(avis_par_entreprise["Titre"], labels=avis_par_entreprise["Entreprise"], autopct="%1.1f%%", startangle=140)
  st.pyplot(fig)
  st.write("*Interprétation :* la majorité des avis collectées (environ 2/3) proviennent d’une même entreprise. Ici, CentralePneus.fr.")

  st.write("##### Evolution de la moyenne des notes dans le temps")
  df["Date_publication"] = pd.to_datetime(df["Date_publication"])
  df["Annee"] = df["Date_publication"].dt.year
  pivot_table = pd.pivot_table(df,values="Note",index=["Annee"], columns="Entreprise", aggfunc="mean")
  pivot_table.index = pd.to_datetime(pivot_table.index, format='%Y')
  fig = plt.figure(figsize=(6, 4))
  sns.set(style="whitegrid")
  sns.lineplot(data=pivot_table, markers=True)
  plt.xlabel("Date de publication")
  plt.ylabel("Moyenne des notes")
  plt.legend(title="Entreprise", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.xticks(rotation=45)
  st.pyplot(fig)
  st.write("*Interprétation :* contrairement à ses concurrents qui ont connu une relative stabilité de leurs notes, la moyenne des notes d’Allopneus a souffert d’une baisse importante en 2013 et 2016.")

  st.write("##### Répartition des notes par entreprise")
  note_counts = df.groupby(["Entreprise","Note"]).size().unstack(fill_value=0)
  ax = note_counts.plot(kind="bar", stacked=True, figsize=(10, 6))
  plt.xlabel("Entreprise")
  plt.ylabel("Nombre d'avis")
  plt.legend(title="Note", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.xticks(rotation=0)
  sns.set_palette("coolwarm")
  fig = plt.gcf()
  st.pyplot(fig)
  st.write("*Interprétation :* ce graphique présente la répartition des notes par entreprises. Il permet de visualiser rapidement la prépondérance d’une note par rapport aux autres.")

  st.write("##### Répartition des réponses par entreprise")
  avis_reponse = df.dropna(subset=["Reponse"])
  fig = plt.figure(figsize=(10, 6))
  ax = sns.violinplot(data=avis_reponse, x="Entreprise", y="Note", inner="point", palette="Set2")
  plt.xlabel("Entreprise")
  plt.ylabel("Note")
  plt.xticks(rotation=45, ha='right', fontsize=10)
  plt.tight_layout()
  st.pyplot(fig)
  st.write("*Interprétation :* on s’aperçoit que contrairement à Pneus Online, les deux autres entreprises répondent uniquement aux avis ayant des notes comprises entre 1 et 3.")

  st.write("##### Répartition du temps moyen de réponse par entreprise")
  df["Date_publication"] = pd.to_datetime(df["Date_publication"])
  df["Date_reponse"] = pd.to_datetime(df["Date_reponse"])
  avis_reponse = df.dropna(subset=["Reponse"])
  avis_reponse["Temps_de_reponse"] = (avis_reponse["Date_reponse"] - avis_reponse["Date_publication"]).dt.days
  temps_moyen_reponse = avis_reponse.groupby("Entreprise")["Temps_de_reponse"].mean().reset_index()
  plt.figure(figsize=(10, 6))
  ax = sns.barplot(data=temps_moyen_reponse, y="Entreprise", x="Temps_de_reponse", orient="h", palette="Set3")
  plt.ylabel("Entreprise")
  plt.xlabel("Temps moyen de réponse")
  plt.yticks(fontsize=10)
  plt.tight_layout()
  fig = plt.gcf()
  st.pyplot(fig)
  st.write("*Interprétation :* a politique de communication varie fortement selon l’entreprise. En effet, on peut voir à travers ce graphique que le temps de réponse moyen est 30 jours pour Pneus Online contre 15 jours pour Allopneus. Quant à CentralePneu.fr, le temps de réponse n’est que d’une journée !.")

  st.write("##### Corrélation entre la variable Note et les autres variables")
  from scipy.stats import pearsonr, chi2_contingency
  import numpy as np
  catVar=['Nombre_avis_publie','Pays','Verification','Entreprise','Date_experience','Date_publication','Note']
  stat =[]
  pval = []
  dofs =[]
  corr=[]
  for col in catVar:
      stat_sub =[]
      pval_sub = []
      dofs_sub =[]
      corr_sub=[]
      for col2 in catVar:
          ct = pd.crosstab(df[col],df[col2])
          conting = chi2_contingency(ct)
          stat_sub.append(conting[0])
          pval_sub.append(conting[1])
          dofs_sub.append(conting[2])
          n=ct.sum().sum()
          corr_sub.append(np.sqrt(conting[0]/(len(df)*(min(ct.shape) - 1))))
      stat.append(stat_sub)
      pval.append(pval_sub)
      dofs.append(dofs_sub)
      corr.append(corr_sub)   
  figcorr = sns.heatmap(np.array(corr),annot=True,cmap='RdBu_r')
  figcorr.set_xticklabels(catVar)
  figcorr.set_yticklabels(catVar)
  figcorr.xaxis.tick_top()
  plt.xticks(rotation=50)
  plt.yticks(rotation=0)
  fig = plt.gcf()
  st.pyplot(fig)
  st.write("*Interprétation :* l’objectif de ce graphique est d’établir une corrélation entre les variables. Pour interpréter ce graphique, il faut comprendre qu’un coefficient de corrélation est compris entre −1 et 1. Plus le coefficient est proche de 1, plus la relation linéaire positive entre les variables est forte. Plus le coefficient est proche de −1, plus la relation linéaire négative entre les variables est forte.")
  st.write("Dans notre cas, on peut observer une corrélation forte entre la note et les colonnes Date_publication et Date_experience (corrélation de 0.89), entre la colonne Date_publication et Verification (corrélation de 0.72) mais également entre les colonnes Date_experience et Date_publication avec la colonne Entreprise (corrélation de 0.64 et 0.65).")

  st.write("##### Répartition des avis en fonction du nombre de caractères dans les titres")
  def get_outliers(serie):
      outLimit_p = serie.quantile(.75) + 1.5*(serie.quantile(.75)-serie.quantile(.25));
      outLimit_n = serie.quantile(.25) - 1.5*(serie.quantile(.75)-serie.quantile(.25));
      print(f'Outliers Limite bas = {outLimit_n}; Outliers Limite haut = {outLimit_p}')
      return outLimit_p, outLimit_n
  df.Titre = df.Titre.astype('str')
  df['longTitre']  = df.Titre.apply(lambda x: len(x))
  print(get_outliers(df.longTitre))
  outLimit_p,outLimit_n  =get_outliers(df.longTitre)
  sns.set_theme()
  plt.figure(figsize=(10,6))
  fig_titre1 = sns.histplot(palette=sns.color_palette("Paired", 5),data=df[(df.longTitre<outLimit_p) & (df.longTitre>outLimit_n)],x='longTitre',hue='Note',multiple='stack')
  fig_titre1.set_xlabel("Nombres de caractères")
  fig_titre1.set_ylabel("Nombre d'avis")
  fig = plt.gcf()
  st.pyplot(fig)
  st.write("*Interprétation :* les notes les plus élevées ont des titres courts tandis que les notes basses présentent majoritairement des titres d’une longueur d'environ 30-40 caractères.")
  
  st.write("##### Répartition des avis en fonction du nombre de caractères dans les commentaires")
  df.Commentaire = df.Commentaire.astype('str')
  df['longCommentaire']  = df.Commentaire.apply(lambda x: len(x))
  outLimit_p,outLimit_n  =get_outliers(df.longCommentaire)
  sns.set_theme()
  plt.figure(figsize=(10,6))
  fig_comment1 = sns.histplot(palette=sns.color_palette('Paired',5),data=df[(df.longCommentaire<outLimit_p) & (df.longCommentaire>outLimit_n)],x='longCommentaire',hue='Note',multiple='stack')
  fig_titre1.set_xlabel("Nombres de caractères")
  fig_titre1.set_ylabel("Nombre d'avis")
  fig = plt.gcf()
  st.pyplot(fig)
  st.write("*Interprétation :* la figure ci-dessus montre que les notes les plus élevées concentrent les commentaires les plus concis tandis que les notes basses possèdent des commentaires de longueur plus variées mais presque toujours avec une longueur minimum de 50 caractères.")

  st.write("##### Analyse des commentaires")
  sns.set_theme()
  f, (ax1, ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10, 15))
  out_p,out_n = get_outliers(df.UpperCount)
  fig001 = sns.histplot(data = df[(df.UpperCount<np.maximum(5,out_p))], x='UpperCount',ax=ax1,discrete=True,hue='Note',palette=sns.color_palette("Paired", 5),multiple='stack')
  fig001.set(title='Répartition du nombre de majuscules')
  fig001.set_xlabel('Nombre de majuscules')
  fig001.set_ylabel('Quantité d\'avis')
  out_p,out_n = get_outliers(df.digitCount)
  fig002 = sns.histplot(data = df[(df.digitCount<np.maximum(5,out_p))], x='digitCount',ax=ax2,discrete=True,hue='Note',palette=sns.color_palette("Paired", 5),multiple='stack')
  fig002.set(title='Répartition du nombre de chiffres')
  fig002.set_xlabel('Nombre de chiffres')
  fig002.set_ylabel('Quantité d\'avis')
  out_p,out_n = get_outliers(df.PonctCount)
  fig003 = sns.histplot(data = df[(df.PonctCount<np.maximum(5,out_p)) ], x='PonctCount',ax=ax3,discrete=True,hue='Note',palette=sns.color_palette("Paired", 5),multiple='stack')
  fig003.set(title='Répartition du nombre de ponctuations')
  fig003.set_xlabel('Nombre de ponctuations (?,!)')
  fig003.set_ylabel('Quantité d\'avis')
  out_p,out_n = get_outliers(df.SpecialCount)
  fig004 = sns.histplot(data = df[(df.SpecialCount<np.maximum(5,out_p))], x='SpecialCount',ax=ax4,discrete=True,hue='Note',palette=sns.color_palette("Paired", 5),multiple='stack')
  fig004.set(title='Répartition du nombre de charactères spéciaux')
  fig004.set_xlabel('Nombre de caractères spéciaux (#$<,etc)')
  fig004.set_ylabel('Quantité d\'avis')
  f.subplots_adjust(hspace=0.4)
  fig = plt.gcf()
  st.pyplot(fig)
  st.write("*Interprétation :* on remarque que sur la plupart des graphiques, les notes les plus élevées ont tendance à avoir un nombre faible de caractères de types majuscules, chiffres, ponctuations et de caractères spéciaux. Néanmoins, il est plus difficile de conclure concernant les notes basses.")
  
  st.write("##### Top 10 des mots/trigrammes les plus présents dans les titres")
  common_words = get_top_ngram(df.loc[df.Note<3]['Commentaire'].dropna().astype(str), n=5,ngrame=1,stopWord=True)
  textCloud=''
  for word, freq in common_words:
      textCloud += ' ' + word 
  df_commonWord = pd.DataFrame(common_words,columns=['mots','count'])
  sns.set_theme()
  ngrameNB=1
  f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(10, 10))
  common_words1 = get_top_ngram(df.loc[df.Note>=3]['Titre'].dropna().astype(str), n=10,ngrame=ngrameNB,stopWord=True)
  df_commonWord1 = pd.DataFrame(common_words1,columns=['mots','count'])
  fig4=sns.barplot(data = df_commonWord1,x='mots',y='count',ax=ax1,palette="deep")
  fig4.set_xticklabels(fig4.get_xticklabels(),rotation=60,ha="right")
  plt.tight_layout()
  fig4.set(title='Avis positifs (Note>=3)')
  ax1.set_ylabel('Nombre d\'occurences')
  ax1.set(xlabel=None)
  common_words2 = get_top_ngram(df.loc[df.Note<3]['Titre'].dropna().astype(str), n=10,ngrame=ngrameNB,stopWord=True)
  df_commonWord2 = pd.DataFrame(common_words2,columns=['mots','count'])
  fig41=sns.barplot(data = df_commonWord2,x='mots',y='count',ax=ax2,palette="deep")
  fig41.set_xticklabels(fig41.get_xticklabels(),rotation=60,ha="right")
  plt.tight_layout()
  fig41.set(title='Avis négatifs (Note<3)')
  ax2.set_ylabel('Nombre d\'occurences')
  ax2.set(xlabel=None)
  ngrameNB=3
  common_words3 = get_top_ngram(df.loc[df.Note>=3]['Titre'].dropna().astype(str), n=10,ngrame=ngrameNB,stopWord=True)
  df_commonWord3 = pd.DataFrame(common_words3,columns=['mots','count'])
  fig42=sns.barplot(data = df_commonWord3,x='mots',y='count',ax=ax3,palette="deep")
  fig42.set_xticklabels(fig42.get_xticklabels(),rotation=60,ha="right")
  plt.tight_layout()
  fig42.set(title='Avis positifs (Note>=3)')
  ax3.set_ylabel('Nombre d\'occurences')
  ax3.set(xlabel=None)
  common_words4 = get_top_ngram(df.loc[df.Note<3]['Titre'].dropna().astype(str), n=10,ngrame=ngrameNB,stopWord=True)
  df_commonWord4 = pd.DataFrame(common_words4,columns=['mots','count'])
  fig43=sns.barplot(data = df_commonWord4,x='mots',y='count',ax=ax4,palette="deep")
  fig43.set_xticklabels(fig43.get_xticklabels(),rotation=60,ha="right")
  plt.tight_layout()
  fig43.set(title='Avis négatifs (Note<3)')
  ax4.set_ylabel('Nombre d\'occurences')
  ax4.set(xlabel=None)
  fig = plt.gcf()
  st.pyplot(fig)

  st.write("##### Top 10 des mots/trigrammes les plus présents dans les commentaires")
  common_words = get_top_ngram(df.loc[df.Note<3]['Commentaire'].dropna().astype(str), n=5,ngrame=1,stopWord=True)
  textCloud=''
  for word, freq in common_words:
      textCloud += ' ' + word 
      print(word, freq) 
  df_commonWord = pd.DataFrame(common_words,columns=['mots','count'])
  sns.set_theme()
  ngrameNB=1
  f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(10, 10))
  common_words1 = get_top_ngram(df.loc[df.Note>=3]['Commentaire'].dropna().astype(str), n=10,ngrame=ngrameNB,stopWord=True)
  df_commonWord1 = pd.DataFrame(common_words1,columns=['mots','count'])
  fig4=sns.barplot(data = df_commonWord1,x='mots',y='count',ax=ax1,palette="deep")
  fig4.set_xticklabels(fig4.get_xticklabels(),rotation=60,ha="right")
  plt.tight_layout()
  fig4.set(title='Avis positifs (Note>=3)')
  ax1.set_ylabel('Nombre d\'occurences')#.ylabel('Nombre d\'occurences')
  ax1.set(xlabel=None)
  common_words2 = get_top_ngram(df.loc[df.Note<3]['Commentaire'].dropna().astype(str), n=10,ngrame=ngrameNB,stopWord=True)
  df_commonWord2 = pd.DataFrame(common_words2,columns=['mots','count'])
  fig41=sns.barplot(data = df_commonWord2,x='mots',y='count',ax=ax2,palette="deep")
  fig41.set_xticklabels(fig41.get_xticklabels(),rotation=60,ha="right")
  plt.tight_layout()
  fig41.set(title='Avis négatifs (Note<3)')
  ax2.set_ylabel('Nombre d\'occurences')#plt.ylabel('Nombre d\'occurences')
  ax2.set(xlabel=None)
  ngrameNB=3
  common_words3 = get_top_ngram(df.loc[df.Note>=3]['Commentaire'].dropna().astype(str), n=10,ngrame=ngrameNB,stopWord=True)
  df_commonWord3 = pd.DataFrame(common_words3,columns=['mots','count'])
  fig42=sns.barplot(data = df_commonWord3,x='mots',y='count',ax=ax3,palette="deep")
  fig42.set_xticklabels(fig42.get_xticklabels(),rotation=60,ha="right")
  plt.tight_layout()
  fig42.set(title='Avis positifs (Note>=3)')
  ax3.set_ylabel('Nombre d\'occurences')
  ax3.set(xlabel=None)
  common_words4 = get_top_ngram(df.loc[df.Note<3]['Commentaire'].dropna().astype(str), n=10,ngrame=ngrameNB,stopWord=True)
  df_commonWord4 = pd.DataFrame(common_words4,columns=['mots','count'])
  fig43=sns.barplot(data = df_commonWord4,x='mots',y='count',ax=ax4,palette="deep")
  fig43.set_xticklabels(fig43.get_xticklabels(),rotation=60,ha="right")
  plt.tight_layout()
  fig43.set(title='Avis négatifs (Note<3)')
  ax4.set_ylabel('Nombre d\'occurences')
  ax4.set(xlabel=None)
  fig = plt.gcf()
  st.pyplot(fig)
  st.write("*Interprétation :* Les commentaires sont une source précieuse d'informations. En conséquence, nous avons utilisé les fonctionnalités offertes par CountVectorizer inclus dans sklearn.feature_extraction.text afin d’extraire les mots ou groupes de mots les plus récurrents dans les commentaires et titres. Afin d'affiner notre étude nous considérons 2 groupes. Les avis positifs présentant une note supérieure ou égale à 3 et les avis négatifs avec une note inférieure à 3. On remarque que les avis positif et négatif présentent une nette prédominance de mots et groupes de mots.")

def demo_model1():
  st.write("### Modélisation")
  # Création de la colonne "Sentiment" à partir de la colonne "Note".
  df2['Sentiment'] = df2['Note'].apply(lambda x: 'Négatif' if x in [1, 2] else 'Positif')
  # Convertir les valeurs de la colonne "Sentiment" en valeurs numériques.
  sentiment_encoder = LabelEncoder()
  df2['Sentiment'] = sentiment_encoder.fit_transform(df2['Sentiment'])
  # Suppression des caractères spéciaux, chiffres, valeurs manquantes et texte en minuscules.
  df2['Titre'] = df2['Titre'].str.replace('[^a-zA-Z\s]', '').str.lower()
  df2['Titre'] = df2['Titre'].fillna('') 
  df2['Commentaire'] = df2['Commentaire'].str.replace('[^a-zA-Z\s]', '').str.lower()
  df2['Commentaire'] = df2['Commentaire'].fillna('') 
  # Suppression des caractères spéciaux, chiffres, valeurs manquantes et texte en minuscules.
  df2['Titre'] = df2['Titre'].str.replace('[^a-zA-Z\s]', '').str.lower()
  df2['Titre'] = df2['Titre'].fillna('') 
  df2['Commentaire'] = df2['Commentaire'].str.replace('[^a-zA-Z\s]', '').str.lower()
  df2['Commentaire'] = df2['Commentaire'].fillna('') 
  # Sélection des colonnes à exclure.
  colonnes_exclues = ['Titre', 'Date_experience', 'Date_publication', 'Reponse', 'Date_reponse', 'Pays', 'Verification', 'Note', 'Entreprise', 'Sentiment']
  # Sélection des colonnes à inclure dans X.
  colonnes_incluses = [colonne for colonne in df2.columns if colonne not in colonnes_exclues]
  # Séparation des données en X (caractéristiques) et y (variable cible).
  X = df2[colonnes_incluses]
  y = df2['Sentiment']
  # Division des données en ensembles d'entraînement et de test.
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  # Extraire la colonne 'Commentaire' de X_train et X_test.
  X_train_commentaire = X_train['Commentaire']
  X_test_commentaire = X_test['Commentaire']
  # Appliquer une vectorisation de texte via CountVectorizer avec stop words en français.
  vectorizer = CountVectorizer(stop_words=list(stop_words_french))
  X_train_commentaire_vectorized = vectorizer.fit_transform(X_train_commentaire)
  X_test_commentaire_vectorized = vectorizer.transform(X_test_commentaire)
  # Convertir les données vectorisées en matrices creuses de type float64.
  X_train_commentaire_vectorized = X_train_commentaire_vectorized.astype('float64')
  X_test_commentaire_vectorized = X_test_commentaire_vectorized.astype('float64')
  # Concaténer les matrices creuses des commentaires avec les autres colonnes.
  X_train_concat = hstack([X_train_commentaire_vectorized] + [X_train.drop('Commentaire', axis=1).values.astype('float64')])
  X_test_concat = hstack([X_test_commentaire_vectorized] + [X_test.drop('Commentaire', axis=1).values.astype('float64')])

  def prediction(classifier):
      if classifier == 'Gradient Boosting':
          clf = joblib.load('gb_clf.joblib')
      elif classifier == 'XGBoost':
          clf = joblib.load('xgb_clf.joblib')
      elif classifier == 'Random Forest':
          clf = joblib.load('rf_clf.joblib')
      elif classifier == 'Logistic Regression':
          clf = joblib.load('lr_clf.joblib')
      elif classifier == 'Support Vector Machine':
          clf = joblib.load('svm_clf.joblib')
      elif classifier == 'Naive Bayes':
          clf = joblib.load('nb_clf.joblib')
      return clf

  def scores(clf, choice):
      if choice == 'Rapport de classification':
          if isinstance(clf, MultinomialNB):
              report = classification_report(y_test, clf.predict(X_test_commentaire_vectorized), output_dict=True)
          else:
              report = classification_report(y_test, clf.predict(X_test_concat), output_dict=True)
          return pd.DataFrame(report).transpose()
      elif choice == 'Matrice de confusion':
          if isinstance(clf, MultinomialNB):
              confusion = confusion_matrix(y_test, clf.predict(X_test_commentaire_vectorized))
          else:
              confusion = confusion_matrix(y_test, clf.predict(X_test_concat))
          confusion_df = pd.DataFrame(confusion, columns=['0','1'], index=['0','1'])
          return confusion_df
      
  choix = ['Gradient Boosting', 'XGBoost', 'Random Forest', 'Logistic Regression', 'Support Vector Machine', 'Naive Bayes']
  option = st.selectbox('Choix du modèle', choix)
  st.write('Le modèle choisi est :', option)

  clf = prediction(option)
  display = st.radio('Que souhaitez-vous montrer ?', ('Rapport de classification', 'Matrice de confusion'))
  if display == 'Rapport de classification':
      st.dataframe(scores(clf, display))
  elif display == 'Matrice de confusion':
      st.dataframe(scores(clf, display))

def demo_model2():

    st.write("### Modelisation II")
    tab_randomForest, tab_XGBoost = st.tabs(["Random Forest", "XGBoost"])

    with tab_randomForest:
        col01, _, col03 = st.columns(3)
        with col01:
            st.write("Hyperparametres:")
        with col03:
            isBest = st.checkbox("Best Random Forest Model")

        col1,col2,col3,col4  = st.columns(4)
        if isBest:
            with col1:
                n = st.selectbox('n_estimator',[100], disabled=True)
            with col2:
                split = st.selectbox('min_samples_leaf',[2], disabled=True)
            with col3:
                leaf = st.selectbox('min_samples_leaf',[1], disabled=True)
            with col4:
                feat = st.selectbox('max_features',["auto"], disabled=True)
        else:
            with col1:
                n = st.selectbox('n_estimator', [100], disabled=True)
            with col2:
                split = st.selectbox('min_samples_leaf', [2, 5, 10])
            with col3:
                leaf = st.selectbox('min_samples_leaf', [1, 5, 10])
            with col4:
                feat = st.selectbox('max_features', ["auto", "sqrt", "log2"])


        with st.spinner(text='Chargement du model'):
            fileName = f'./dataBase_models/modelisation2_randFor_n_{n}_minsamplessplit_{split}_minsamplesleaf' \
                       f'_{leaf}_maxfeatures_{feat}.pkl'
            loaded_model = get_saved_model(fileName)

            x_test = get_df("./xTest_embedded.csv")
            y_test = get_df("./yTest_embedded.csv")
            y_pred = loaded_model.predict(x_test)
            y_pred_prob = loaded_model.predict_proba(x_test)


        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())


        classification_plots(y_test.Sentiment.tolist(),y_pred_prob)


        X_train = get_df('xTrain_embedded.csv')
        X_train = X_train.sample(100)
        explainer = shap.Explainer(loaded_model)
        shap_values = explainer.shap_values(X_train)

        plt.figure()
        shap.summary_plot(shap_values, features=sample_data, feature_names=X_train.columns)
        st.pyplot(plt.gcf())


    with tab_XGBoost:
        col01, col02, col03 = st.columns(3)
        with col01:
            st.write("Hyperparametres:")
        with col03:
            isBest = st.checkbox('Best XGBoost Model')

        col1, col2, col3, col4 = st.columns(4)
        if isBest:
            with col1:
                n = st.selectbox('max_depth', [7], disabled=True)
            with col2:
                split = st.selectbox('learning_rate', [0.2], disabled=True)
            with col3:
                leaf = st.selectbox('n_estimators', [200], disabled=True)
            with col4:
                evalMetric = st.selectbox('eval_metric', ['logloss'], disabled=True)

        else:
            with col1:
                maxDepth = st.selectbox('max_depth', [5,7])
            with col2:
                learnRate = st.selectbox('learning_rate', [0.1, 0.2])
            with col3:
                nEstim = st.selectbox('n_estimators', [100,200])
            with col4:
                evalMetric = st.selectbox('eval_metric', ['logloss','error','aucpr'])


        with st.spinner(text='Chargement du model'):
            fileName = f'./dataBase_models/modelisation2_xgBoost_learningRate_{learnRate}_maxDepth_' \
                       f'{maxDepth}_nEstimators' \
                       f'_{nEstim}_evalMetric' \
                       f'_{evalMetric}.pkl'
            loaded_model = get_saved_model(fileName)

            x_test = get_df("./xTest_embedded.csv")
            y_test = get_df("./yTest_embedded.csv")
            y_pred = loaded_model.predict(x_test)
            y_pred_prob = loaded_model.predict_proba(x_test)

        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

        classification_plots(y_test.Sentiment.tolist(),y_pred_prob)

def demo_model3():
    st.write("### Modelisation III")

    tab_1, tab_2 = st.tabs(["architecure simple", "architecture 2"])

    with tab_1:
        col01, _, _ = st.columns(3)

        with st.spinner(text='Chargement du model'):
            fileName = f'./dataBase_models/modelisation3_simpleNeuralNetwork.h5'
            loaded_model = load_model(fileName)

            x_test = get_df("./xTest_embedded.csv")
            y_test = get_df("./yTest_embedded.csv")
            y_pred = loaded_model.predict(x_test)
            # y_pred_prob = loaded_model.predict(x_test)

        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred[:,1]>0.5, output_dict=True)).transpose())

        classification_plots(y_test.Sentiment.tolist(),y_pred)

def demo_interact():
    # get_fastTextModel()
    loadedModel = get_saved_model('./dataBase_models/modelisation2_xgBoost_learningRate_0.2_maxDepth_7_nEstimators_200_evalMetric_logloss.pkl')

    st.write("### Interactivite")

    # Display a chat input widget.
    titre = st.text_input("Laissez un titre")
    comment = st.text_area("Laissez un commentaire")

    if titre != "" and comment != "":
        df=pd.DataFrame()
        dfComment = pd.DataFrame([get_text_embedding(comment)])
        dfComment = dfComment.add_prefix('vectComment_')

        dfTitre = pd.DataFrame([get_text_embedding(comment)])
        dfTitre = dfTitre.add_prefix('vectTitre_')

        st.dataframe(dfComment)


        df =  pd.concat([dfComment,dfTitre],axis=1)
        df['Nombre_avis_publie'] = 0
        df['nb_majuscules_Titre'], _, df['nb_chiffres_Titre'], df['nb_ponctuation_Titre'], df['nb_special_Titre'] = Count_special(
            titre)
        df['nb_majuscules_Commentaire'], _, df['nb_chiffres_Commentaire'], df['nb_ponctuation_Commentaire'], df['nb_special_Commentaire'] = Count_special(
            comment)
        df['longCommentaire'] = len(comment)
        df['longTitres'] = len(titre)
        df['nb_Mots_Commentaire'] = len(comment.split(' '))
        df['nb_Mots_Titres'] = len(titre.split(' '))

        # df.reset_index(drop=True, inplace=True)
        # df = df.drop([0])

        df_test = df.filter(like='vect')

        df = pd.concat([df_test, df[
            ['Nombre_avis_publie', 'longCommentaire', 'longTitres', 'nb_Mots_Commentaire', 'nb_Mots_Titres',
             'nb_majuscules_Commentaire',
             'nb_chiffres_Commentaire', 'nb_ponctuation_Commentaire',
             'nb_special_Commentaire', 'nb_majuscules_Titre', 'nb_chiffres_Titre',
             'nb_ponctuation_Titre', 'nb_special_Titre']]], axis=1)

    targetType = st.selectbox('Quel type de prediction ?',['Sentiment', 'Note'])

    if targetType == 'Sentiment' and titre!="" and comment != "":
        # y_pred = loadedModel.predict(data)
        y_pred = loadedModel.predict(df)
        if y_pred:
            # rainningEmoji("😄")
            st.write('Commentaire sympas')
            left_co, cent_co, last_co = st.columns(3)
            with cent_co:
                st.image("emojiHappy.webp")
        else:
            # rainningEmoji("😱")
            st.write('Pas content')
            left_co, cent_co, last_co = st.columns(3)
            with cent_co:
                st.image("emojiSad.png")


#############################################################################################################
#                           Fonction Principale
#############################################################################################################
def demo_supplyChain():

    #Chargement donnees dans le cache:



    #Side bar - Sommaire:

    st.title("Projet Supply Chain")
    st.header("Analyse de sentiments sur Pneus")
    st.sidebar.title("Sommaire")
    pages = ["Introduction","Exploration", "Representation Graphique", "Modélisation I","Modélisation II",
             "Modélisation III",
             "Interactivite"]
    page = st.sidebar.radio("Aller vers", pages)

    match page:
        case "Introduction":
            demo_intro()

        case "Exploration":
            demo_explor()

        case "Representation Graphique":
            demo_reprGraph()

        case "Modélisation I":
            demo_model1()

        case "Modélisation II":
            demo_model2()

        case "Modélisation III":
            demo_model3()

        case "Interactivite":
            demo_interact()

        case _:
            st.write("ERREUR: PAGE INTROUVABLE")

demo_supplyChain()