import pandas as pd
from pyproj import Proj, transform

# Charger le fichier CSV dans un DataFrame pandas
data = pd.read_csv('../Fichiers pour challenge 3/Mesures sur 41 45 89.csv', delimiter=';')

# Extraire les coordonnées x et y du DataFrame
x = data['X']
y = data['Y']

# Définir les systèmes de coordonnées de départ (Lambert 93) et d'arrivée (WGS84, utilisé pour les latitudes et longitudes)
lambert93 = Proj(init='epsg:2154') # Lambert 93
wgs84 = Proj(init='epsg:4326') # WGS84 (lat/lon)

# Convertir les coordonnées
lon, lat = transform(lambert93, wgs84, x, y)

# Ajouter les colonnes lon et lat au DataFrame
data['lon'] = lon
data['lat'] = lat

# Enregistrer le DataFrame avec les nouvelles colonnes dans un nouveau fichier CSV
data.to_csv('data_with_lon_lat.csv', sep=';', index=False)
