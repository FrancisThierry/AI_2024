import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger le modèle depuis le fichier
with open('modele_regression_uk_sales.pkl', 'rb') as fichier:
    model = pickle.load(fichier)

# Créer un DataFrame pour la prédiction
new_data = pd.DataFrame({
    'Year': [1995],
    'Region_Name_City of London': [1],
    'Region_Name_Inner London': [0],
    'Region_Name_London': [0],
    'Region_Name_Outer London': [0],
})

# Normaliser les données en utilisant le même scaler que celui utilisé lors de l'entraînement
scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(new_data)

# Prédire le volume de vente pour la région City of London en 1995
predicted_volume = model.predict(new_data_scaled)

print(f'La prédiction du volume de vente pour 1995 à London City est : {predicted_volume[0]}')
