import pycaret
from pycaret.datasets import get_data

# Charger le jeu de données diabetes
data = get_data('diabetes')

# Initialiser PyCaret pour la classification binaire
from pycaret.classification import *
clf = setup(data, target='Class variable')

# Créer le modèle de classification (dans cet exemple, nous utilisons Random Forest)
rf = create_model('rf')

# Tracer le diagramme pour montrer la classification
plot_model(rf, plot='boundary')
