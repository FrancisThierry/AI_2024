import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Exemple de dataset fictif avec noms de fruits
# Chaque ligne représente [nom du fruit, teneur en vitamine C, acidité, diamètre, provenance (0 pour non agrume, 1 pour agrume)]
dataset = np.array([
    ['Orange', 15, 2.5, 7, 1],
    ['Apple', 7, 3, 6, 0],
    ['Lemon', 10, 2, 5, 1],
    ['Grape', 8, 4, 4, 0],
    ['Banana', 6, 2, 3, 0],
    ['Lime', 9, 3, 4, 1],
    ['Pear', 5, 4, 5, 0],
    ['Grapefruit', 11, 1, 6, 1],
    ['Cherry', 4, 3.5, 2, 0],
    ['Strawberry', 12, 4, 3, 1],
    ['Peach', 3, 2.5, 4, 0],
    ['Mandarin', 13, 1.5, 3, 1],
    ['Plum', 2, 3, 2, 0],
    ['Kiwi', 14, 2, 5, 1],
    ['Pineapple', 1, 4, 7, 0]
])

# Séparation des features (X) et des labels (y)
X = dataset[:, 1:4].astype(float)  # Utilisation des colonnes 1 à 3 pour les features (vitamine C, acidité, diamètre)
y = dataset[:, 4].astype(int)

# Création du classifieur SVM
clf = svm.SVC(kernel='linear')

# Entraînement du modèle
clf.fit(X, y)

# Affichage de la séparation
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

# Tracé des vecteurs de support
sv = clf.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], marker='x', color='red', s=100, label='Support Vectors')

# Dessin de la frontière de décision
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 15)
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-', label='Decision Boundary')

# Affichage des noms des fruits
for i, fruit in enumerate(dataset):
    plt.annotate(fruit[0], (X[i, 0], X[i, 1]))

plt.xlabel('Teneur en vitamine C')
plt.ylabel('Acidité')
plt.title('Classification SVM de fruits')
plt.legend()
plt.show()
