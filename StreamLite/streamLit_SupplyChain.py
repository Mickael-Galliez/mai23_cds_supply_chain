#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:07:36 2022

@author: yohancohen
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gensim.models import FastText


# Téléchargez la liste de stop words en français
nltk.download('stopwords')
nltk.download('punkt')

# Importez la liste de stop words en français
from nltk.corpus import stopwords

@st.cache_resource
def get_df():
    """
    charge le dataframe et applique les dernieres etapes de nettoyage.
    doit etre applique 1 fois au debut de la fonction principale pour charge le dataframe dans le cache.

    :return: dataframe
    """

    df = pd.read_csv("df_featEngin.csv")

    return df
@st.cache_resource
def get_dfFastText():
    """
    Retourne un dataframe avec les colonnes commentaires et titres vectorises.
    Cette fonction est utile pour les sections modelisation II et III.
    On appelle la fonction en debut de fonction principale pour charger les donnees dans le cache
    :return:
    """

    df = get_df()

    df['embedding_Commentaire'] = df['Commentaire'].apply(get_text_embedding)
    df['embedding_Titres'] = df['Titres'].apply(get_text_embedding)

    df_expanded = df_test['embedding_Commentaire'].apply(lambda x: pd.Series(x, dtype="float64")).add_prefix(
        'vectComment_')
    df_expanded2 = df_test['embedding_Titres'].apply(lambda x: pd.Series(x, dtype="float64")).add_prefix('vectTitre_')

    df = pd.concat([df_expanded,df_expanded2,df[['Nombre_avis_publie','longCommentaire','longTitres','nb_Mots_Commentaire',
        'nb_Mots_Titres', 'nb_majuscules_Commentaire',
           'nb_chiffres_Commentaire', 'nb_ponctuation_Commentaire',
           'nb_special_Commentaire', 'nb_majuscules_Titre', 'nb_chiffres_Titre',
           'nb_ponctuation_Titre', 'nb_special_Titre', 'Sentiment']]],axis=1)


@st.cache_resource
def get_fastTextModel():
    """
    retourne le model fast text. Ce model est long a construire aussi nous utiliserons le decorateur cache ressource
    cette fonction doit etre appele une premiere fois au debut de la fonction principale afin de charge le model dans le cache
    elle pourra etre appele a nouveau par la suite dans les differentes sections sans temps de chargement.
    :return: model fast text
    """
    # Replace 'path/to/pretrained/embeddings' with the actual path to your downloaded embeddings file
    return FastText.load_fasttext_format("FastText/cc.fr.300.bin.gz")

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



#############################################################################################################
#                           Fonctions de Sections
#############################################################################################################
def demo_intro():
    """
    Definit la section Introduction du rapport
    :return: None
    """

    st.write("### Introduction")
    st.image("./wordCloud.png")

    st.write("Notre choix s'est porte sur l'etude d'avis client concernant les pneus. "
             "L'enjeux de cet exercice etait la mise au point d'un modele efficace de "
             "Sentiment analysis permettant de predire si un avis/commentaire se revellait "
             "Positif ou Negatif.\n")

def demo_explor():
    df = get_df()

    st.write("### Exploration")
    st.write("#### WebScrapping")
    st.write("....")
    st.write("#### Etude Preliminaire:")
    st.dataframe(df[["Nombre_avis_publie","Verifications","Titres","Commentaire","Saison_experience", "Note"]])


def demo_reprGraph():
    st.write("### Representation Graphique")
    st.write("### Visualisation des données")



def demo_model1():
    st.write("### Modelisation I")

def demo_model2():
    st.write("### Modelisation II")
    tab1, tab2 = st.tabs(["Random Forest", "XGBoost"])
    with tab1:
        col01, col02, col03 = st.columns(3)
        with col01:
            st.write("Hyperparametres:")
        with col03:
            isBest = st.checkbox('Selection automatique')

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
                n = st.selectbox('n_estimator', [100, 200, 500])
            with col2:
                split = st.selectbox('min_samples_leaf', [2, 5, 10])
            with col3:
                leaf = st.selectbox('min_samples_leaf', [1, 5, 10])
            with col4:
                feat = st.selectbox('max_features', ["auto", "sqrt", "log2"])


        with st.spinner(text='Chargement du model'):
            fileName = f'./dataBase_models/modelisation2_randFor_n_{n}_minsamplessplit_{split}_minsamplesleaf' \
                       f'_{leaf}_maxfeatures_{feat}.pkl'
            loaded_model = joblib.load(fileName)

            df = get_dfFastText()
            _, x_test,_,y_test = get_splited_df(df.drop(['Sentiment'],df['Sentiment']))
            y_pred = loaded_model.predict(x_test)

        st.dataframe(classification_report(y_test, y_pred, output_dict=True).transpose())
def demo_interact():
    st.write("### Interactivite")

    with st.chat_message("user"):
        st.write("Hello 👋")
        st.line_chart(np.random.randn(30, 3))

    # Display a chat input widget.
    input = st.chat_input("Say something")

    st.write(input)



#############################################################################################################
#                           Fonction Principale
#############################################################################################################
def demo_supplyChain():

    #Chargement donnees dans le cache:
    get_df()
    #get_fastTextModel() # Cette etape necessitte 5 a 10 min de chargement.
    #get_dfFastText()    # L'etape precedente est necessaire pour effectue celle ci

    #Side bar - Sommaire:

    st.title("Projet Supply Chain")
    st.header("Analyse de sentiments sur Pneus")
    st.sidebar.title("Sommaire")
    pages = ["Introduction","Exploration", "Representation Graphique", "Modélisation I","Modélisation II",
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

        case "Interactivite":
            demo_interact()

        case _:
            st.write("ERREUR: PAGE INTROUVABLE")






demo_supplyChain()
