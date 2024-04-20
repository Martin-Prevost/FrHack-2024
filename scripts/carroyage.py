import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Charger les données depuis un fichier CSV
data = np.genfromtxt('data.csv', delimiter=';', dtype=None, names=True, encoding=None)

x = data['X']
y = data['Y']
puissance_dbm = data['dbm']

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

# Définir la taille de la grille et le nombre de cellules
grid_size = 100
x_grid = np.linspace(x_min, x_max, grid_size + 1)  # Ajoutez 1 à grid_size pour avoir le bon nombre de cellules
y_grid = np.linspace(y_min, y_max, grid_size + 1)

# Initialiser la matrice pour stocker les moyennes
average_power = np.full((grid_size, grid_size), np.nan)

grille = []

# Calculer la moyenne des points dans chaque cellule de la grille
for i in range(grid_size):
    for j in range(grid_size):
        x_lower, x_upper = x_grid[i], x_grid[i + 1]
        y_lower, y_upper = y_grid[j], y_grid[j + 1]
        indices = np.where((x >= x_lower) & (x < x_upper) & (y >= y_lower) & (y < y_upper))
        if len(indices[0]) > 0:
            average_power[i, j] = np.mean(puissance_dbm[indices])
            grille.append([(), x_upper, y_lower, y_upper, list(indices), average_power[i, j]])
        else:
            grille.append([x_lower, x_upper, y_lower, y_upper, list(indices), 0])


# Créer la grille de points
X, Y = np.meshgrid(x_grid[:-1], y_grid[:-1])

# Tracer la grille de points avec le gradient de couleur en fonction de la puissance moyenne
plt.pcolormesh(X, Y, average_power.T, cmap='RdYlGn', shading='auto', norm=Normalize(vmin=np.nanmin(average_power), vmax=np.nanmax(average_power)), edgecolor='face')

# Ajouter des légendes et titres
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Puissance moyenne (dBm)')
plt.title('Carroyage de la puissance moyenne (dBm)')

# Afficher le graphique
plt.show()
