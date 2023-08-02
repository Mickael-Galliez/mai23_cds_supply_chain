
###################################
## 2 - Data pre-processing

import locale
import pandas as pd


fileName = 'Df_alloPneus'
pathName = './Data'

# Importer le DataFrame consolide.
df_Allopneus = pd.read_csv(f'{pathName}\{fileName}_brut.csv')
print(df_Allopneus.head())
print(df_Allopneus.info())

# Transformer les colonnes.
df_Allopneus["Nombre_avis_publie"] = df_Allopneus["Nombre_avis_publie"].str.replace(' avis', '')
df_Allopneus["Note"] = df_Allopneus["Note"].str.extract(r'(\d+)')
locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
df_Allopneus["Date_experience"] = df_Allopneus["Date_experience"].str.replace("Date de l'expérience: ","")


df_Allopneus["Date_experience"] = pd.to_datetime(df_Allopneus["Date_experience"], format='ISO8601',errors='coerce')#'%d %m %Y',errors='coerce')
df_Allopneus["Date_experience"] = df_Allopneus["Date_experience"].dt.strftime('%Y-%m-%d')
locale.setlocale(locale.LC_TIME, '')
df_Allopneus["Date_publication"] = pd.to_datetime(df_Allopneus["Date_publication"],
                                                     format='%Y-%m-%dT%H:%M:%S.%fZ').dt.date
df_Allopneus["Date_reponse"] = pd.to_datetime(df_Allopneus["Date_reponse"],
                                                 format='%Y-%m-%dT%H:%M:%S.%fZ').dt.date

# Exportation du DataFrame dans un fichier Excel.
df_Allopneus.to_csv(rf'{pathName}\{fileName}_clean.csv', index=False)
