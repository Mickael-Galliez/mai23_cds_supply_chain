#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:07:36 2022

@author: yohancohen
"""

import streamlit as st
import pandas as pd

def demo_intro():
    st.write("### Introduction")
    st.image("./wordCloud.png")

    st.write("Notre choix s'est porte sur l'etude d'avis client concernant les pneus. "
             "L'enjeux de cet exercice etait la mise au point d'un modele efficace de "
             "Sentiment analysis permettant de predire si un avis/commentaire se revellait "
             "Positif ou Negatif.\n")

def demo_explor():
    df = pd.read_csv("df_featEngin.csv")

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

def demo_interact():
    st.write("### Interactivite")


def demo_supplyChain():

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