# coding: utf-8

# Implémentation de K-means clustering python
# Code produit par le site https://mrmint.fr

#Chargement des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets


#chargement de jeu des données Iris
iris = datasets.load_iris()

#importer le jeu de données Iris dataset à l'aide du module pandas
x = pd.DataFrame(iris.data)

x.columns = ['Sepal_Length','Sepal_width','Petal_Length','Petal_width']

y = pd.DataFrame(iris.target)


y.columns = ['Targets']


#Création d'un objet K-Means avec un regroupement en 3 clusters (groupes)
model=KMeans(n_clusters=3)



#application du modèle sur notre jeu de données Iris
model.fit(x)


colormap=np.array(['Red','green','blue'])

# Initialisation de la liste des inerties
inertia = []

# Calcul de l'inertie pour différentes valeurs de k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)

# Tracé du graphique pour visualiser la méthode du coude
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour le nombre optimal de clusters')
plt.xticks(range(1, 11))
plt.show()