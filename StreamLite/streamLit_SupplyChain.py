#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:07:36 2024

@author: Asselot Joan && Mickael Galliez
"""

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
    df = get_fulldf()

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