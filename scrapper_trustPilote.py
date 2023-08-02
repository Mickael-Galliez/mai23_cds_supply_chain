#####################################
## 1 - Collecte des donnees

# Importation des librairies necessaires.
import os
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import datetime as dt
import lxml
import time
import numpy as np

# Nom du fichier excel contenant les donnees recoltees
fileName = 'Df_alloPneus'
pathName = './Data'

#Verifions que nous necrasons pas de donnees pre-existantes:
if os.path.exists(f'{pathName}\{fileName}_brut.csv'):
    raise Exception(f"Le fichier {pathName}\{fileName}_brut.csv existe deja. Vous risquez decraser vos donnees. Changez de nom de dossier ou supprimez le fichier pre-existant")

# Creation de l'objet BeautifulSoup dans une variable nommee soup.
url = "https://fr.trustpilot.com/review/www.allopneus.com"
from_page = 1
to_page = 10#2970
nom_clients, nombre_avis_publies, notes, titres, commentaires, date_experiences, date_publications, localisations, reponses, date_reponses, pays, verified_status =[], [], [], [], [], [], [], [], [], [], [], []

page = requests.get(url)
soup = bs(page.content, "lxml")

#
# RECUPERATION DES INFORMATIONS CLEFS:

nom_entreprise = soup.find('span', attrs={
    'class': "typography_display-s__qOjh6 typography_appearance-default__AAY17 title_displayName__TtDDM"}).text
nombre_avis = soup.find('p', attrs={'class': "typography_body-l__KUYFJ typography_appearance-default__AAY17"}).text
moyenne_note = soup.find('span',
                         attrs={'class': "typography_heading-m__T_L_X typography_appearance-default__AAY17"}).text

print("Nom de l'entreprise :", nom_entreprise)
print("Nombre d'avis :", nombre_avis)
print("Moyenne des notes :", moyenne_note)

#
# COLLECTE DES AVIS:

for page in range(from_page, to_page + 1):
    url_page = f'{url}?page={page}'

    try:
        response = requests.get(url_page)
    except Exception as e:
        print(repr(e))
        break

    soup = bs(response.content, 'lxml')
    avis_client = soup.find_all('div', attrs={'class': "styles_reviewCardInner__EwDq2"})

    for avis in avis_client:
        nom_client_element = avis.find('span',
                                       class_='typography_heading-xxs__QKBS8 typography_appearance-default__AAY17')
        nom_client = nom_client_element.text.strip() if nom_client_element else None
        nom_clients.append(nom_client)
        nombre_avis_publie = avis.find('span',
                                       class_='typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l').text.strip()
        nombre_avis_publies.append(nombre_avis_publie)
        note = avis.find(class_="star-rating_starRating__4rrcf star-rating_medium__iN6Ty").findChild()
        notes.append(note["alt"])
        titre = avis.find('h2', class_='typography_heading-s__f7029 typography_appearance-default__AAY17').text.strip()
        titres.append(titre)
        commentaire_element = avis.find('p',
                                        class_='typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn')
        commentaire = commentaire_element.text.strip() if commentaire_element else None
        commentaires.append(commentaire)
        date_experience_element = avis.find('p', class_='typography_body-m__xgxZ_ typography_appearance-default__AAY17')
        date_experience = date_experience_element.text.strip() if date_experience_element else None
        date_experiences.append(date_experience)
        date_publication_element = avis.find('time', class_="")
        pays_element = avis.find(lambda tag: tag.name=='svg' and tag.get('class') == ['icon_icon__ECGRl'])
        verif_element = avis.find(lambda tag: tag.name=='svg' and tag.get('class') == ['icon_icon__ECGRl']).find_next('svg') if avis.find(lambda tag: tag.name=='svg' and tag.get('class') == ['icon_icon__ECGRl']) else None

        if date_publication_element:
            date_publication = date_publication_element.get('datetime')
            date_publications.append(date_publication)
        else:
            date_publications.append(None)
        reponse_element = avis.find('p',
                                    class_='typography_body-m__xgxZ_ typography_appearance-default__AAY17 styles_message__shHhX')
        if reponse_element:
            reponse = reponse_element.text.strip()
            reponses.append(reponse)
        else:
            reponses.append(None)
        date_reponse_element = avis.find('time',
                                         class_='typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l styles_replyDate__Iem0_')
        if date_reponse_element:
            date_reponse = date_reponse_element.get('datetime')
            date_reponses.append(date_reponse)
        else:
            date_reponses.append(None)

        if pays_element:
            pays.append(pays_element.find_next('span').text.strip())
        else:
            pays.append(None)

        if verif_element:
            verified_status.append(1) if verif_element.find_next('span').text.strip() == 'Vérifié' else verified_status.append(0)
        else:
            pays.append(0)


    #Temporisation: attendre avant la prochaine requete html
    duree_temporisation = np.random.randint(low=4, high=10)#*0.2
    time.sleep(duree_temporisation)
    
    #Informations reporte apres extraction de donnees de chaque page:
    print(f'page {page} scrapper, temporisation = {duree_temporisation} s, nombre avis recupere: {len(reponses)}')


#
# INTEGRATION DES DONNEES DANS UN DATAFRAME

df_1 = pd.DataFrame(list(
    zip(nom_clients, nombre_avis_publies, notes, titres, commentaires, date_experiences, date_publications, reponses,
        date_reponses,pays,verified_status)),
                    columns=["Client", "Nombre_avis_publie", "Note", "Titres", "Commentaire", "Date_experience",
                             "Date_publication", "Reponse", "Date_reponse","pays",'Avis_verifie'])

# Ajout du nom de l'entreprise evaluee dans nouvelle colonne.
df_1["Entreprise"] = nom_entreprise
df_1.info()

# Exportation du DataFrame dans un fichier Excel.
df_1.to_csv(fr'{pathName}\{fileName}_brut.csv', index=False)
