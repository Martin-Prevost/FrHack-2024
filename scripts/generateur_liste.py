import numpy as np
import json
from pyproj import Proj, transform
import pickle

data = np.genfromtxt('data.csv', delimiter=';', dtype=str, skip_header=1)

ids = data[:, 0]
x = data[:, 1].astype(int)
y = data[:, 2].astype(int)
technos = data[:, 3]
operateurs = data[:, 4]
dbms = data[:, 5].astype(int)
puissances_recues = data[:, 6]

lambert93 = Proj(init='epsg:2154') # Lambert 93
wgs84 = Proj(init='epsg:4326') # WGS84 (lat/lon)

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

# Définir la taille de la grille et le nombre de cellules
pas = 500
x_grid = np.arange(x_min, x_max + pas, pas)  # Ajoutez 1 à grid_size pour avoir le bon nombre de cellules
y_grid = np.arange(y_min, y_max + pas, pas)

print("longueur x_grid", len(x_grid))
print("longueur y_grid", len(y_grid))

grille = []
centres_lambert = []
sommets1_lambert = []
sommets2_lambert = []
sommets3_lambert = []
sommets4_lambert = []

# Calculer la moyenne des points dans chaque cellule de la grille
for i in range(len(x_grid)-1):
    for j in range(len(y_grid)-1):
        x_lower, x_upper = float(x_grid[i]), float(x_grid[i + 1])
        y_lower, y_upper = float(y_grid[j]), float(y_grid[j + 1])
        indices = np.where((x >= x_lower) & (x < x_upper) & (y >= y_lower) & (y < y_upper))
        centre_lambert = (x_lower + x_upper) / 2, (y_lower + y_upper) / 2
        sommet1_lambert = (x_lower, y_lower)
        sommet2_lambert = (x_upper, y_lower)
        sommet3_lambert = (x_upper, y_upper)
        sommet4_lambert = (x_lower, y_upper)
        centres_lambert.append(centre_lambert)
        sommets1_lambert.append(sommet1_lambert)
        sommets2_lambert.append(sommet2_lambert)
        sommets3_lambert.append(sommet3_lambert)
        sommets4_lambert.append(sommet4_lambert)
        if len(indices[0]) > 0:
            average_power = np.mean(dbms[indices])
            grille.append({
                'c_l':centre_lambert,
                's1_l': sommet1_lambert,
                's2_l': sommet2_lambert,
                's3_l': sommet3_lambert,
                's4_l': sommet4_lambert,
                'releves': len(indices[0]),
                'dbm_moy': float(average_power),
                'type': None})
        else:
            grille.append({
                'c_l':centre_lambert,
                's1_l': sommet1_lambert,
                's2_l': sommet2_lambert,
                's3_l': sommet3_lambert,
                's4_l': sommet4_lambert,
                'releves': 0,
                'dbm_moy': 0,
                'type': None})            

x_centres_gps, y_centres_gps = transform(lambert93, wgs84, np.array(centres_lambert)[:, 0], np.array(centres_lambert)[:,1])
x_sommets1_gps, y_sommets1_gps = transform(lambert93, wgs84, np.array(sommets1_lambert)[:, 0], np.array(sommets1_lambert)[:,1])
x_sommets2_gps, y_sommets2_gps = transform(lambert93, wgs84, np.array(sommets2_lambert)[:, 0], np.array(sommets2_lambert)[:,1])
x_sommets3_gps, y_sommets3_gps = transform(lambert93, wgs84, np.array(sommets3_lambert)[:, 0], np.array(sommets3_lambert)[:,1])
x_sommets4_gps, y_sommets4_gps = transform(lambert93, wgs84, np.array(sommets4_lambert)[:, 0], np.array(sommets4_lambert)[:,1])
print(len(x_centres_gps))
for i in range(len(x_centres_gps)):
    grille[i]['centre_gps'] = (float(x_centres_gps[i]), float(y_centres_gps[i]))
    grille[i]['s1_gps'] = (float(x_sommets1_gps[i]), float(y_sommets1_gps[i]))
    grille[i]['s2_gps'] = (float(x_sommets2_gps[i]), float(y_sommets2_gps[i]))
    grille[i]['s3_gps'] = (float(x_sommets3_gps[i]), float(y_sommets3_gps[i]))
    grille[i]['s4_gps'] = (float(x_sommets4_gps[i]), float(y_sommets4_gps[i]))
    
with open("grille.pkl", "wb") as fichier:
    pickle.dump({'grille': grille,
                 'pas': 100,
                 'len_x_grid': len(x_grid),
                 'len_y_grid': len(y_grid),}, fichier)