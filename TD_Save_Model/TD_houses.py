import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
# Enregistrement du modèle avec Pickle

# Création du dataframe
df = pd.DataFrame({
    'surface': [100, 120, 150, 180],
    'nb_pieces': [3, 4, 5, 6],
    'localisation': ['ville', 'campagne', 'ville', 'campagne'],
    'prix': [300000, 200000, 450000, 250000]
})

# Encodage de la variable catégorielle "localisation"
df = pd.get_dummies(df, columns=['localisation'])

# Création du modèle de régression linéaire
model = LinearRegression()

with open('modele_regression.pkl', 'wb') as fichier:
    pickle.dump(model, fichier)

# Entrainement du modèle
model.fit(df[['surface', 'nb_pieces', 'localisation_ville']], df['prix'])

# Prédiction du prix pour une nouvelle maison
surface_nouvelle_maison = 140
nb_pieces_nouvelle_maison = 4
localisation_nouvelle_maison = 'ville'

prediction = model.predict([[surface_nouvelle_maison, nb_pieces_nouvelle_maison, 1]])

# Affichage de la prédiction
print(f"Prix prédit pour une maison de {surface_nouvelle_maison}m² avec {nb_pieces_nouvelle_maison} pièces en {localisation_nouvelle_maison}: {prediction[0]}")
