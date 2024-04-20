import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Charger les données depuis un fichier CSV
data = np.genfromtxt('data.csv', delimiter=';', dtype=None, names=True, encoding=None)

x = data['X']
y = data['Y']
puissance_dbm = data['dbm']

# Créer un tableau de features en combinant x et y
features = np.column_stack((x, y))

# Initialiser le modèle de clustering (K-means par exemple)
num_clusters = 200
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Adapter le modèle aux données
kmeans.fit(features)

# Assigner chaque point à un cluster
cluster_labels = kmeans.predict(features)

# Limiter la plage de valeurs de x et y
x_min, x_max = min(x), max(x)
y_min, y_max = min(y), max(y)

# Définir le nombre de points à échantillonner pour la grille
num_samples = 1000

# Échantillonner aléatoirement un sous-ensemble de données pour la grille
x_samples = np.random.choice(x, num_samples)
y_samples = np.random.choice(y, num_samples)

# Créer une grille avec un incrément de 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, num_samples),
                     np.linspace(y_min, y_max, num_samples))

# Prédire le cluster pour chaque point de la grille
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Remodeler le résultat dans la forme de la grille
Z = Z.reshape(xx.shape)

# Afficher les zones de chaque cluster
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8)
#plt.scatter(x, y, c=cluster_labels, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Zones de chaque cluster')
plt.colorbar()
plt.show()
