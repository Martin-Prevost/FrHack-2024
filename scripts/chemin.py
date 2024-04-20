import numpy as np
from pyproj import Proj, transform
import pickle
from shapely.geometry import Polygon, Point
import pandas as pd
import shapely.geometry as sg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import geopandas as gpd
from alive_progress import alive_bar
import os
import time
import math

start_time = time.time()

filename = "data/Mesures sur 41 45 89.csv"
dept_file = "data/Shape Depts 41 45 89.shp"
town_file = "data/Shape Blois Orleans Auxerre.shp"
peri_urbaines_file = "data/Zones PERI URBAINES 41 45 89.shp"
rurales_file = "data/Zones RURALES 41 45 89.shp"
urbaines_file = "data/Zones URBAINES 41 45 89.shp"

data = np.genfromtxt(filename, delimiter=';', dtype=str, skip_header=1)

size_urb = 1500
selected_operator = "OP1"
techno_list = ["4G", "5G", "all"]
selected_techno = techno_list[2]
nb_valeur_moy = 1

if selected_techno != "all":
    data = data[data[:, 3] == selected_techno]

data = data[data[:, 4] == selected_operator]

ids = data[:, 0]
x = data[:, 1].astype(int)
y = data[:, 2].astype(int)
technos = data[:, 3]
operateurs = data[:, 4]
dbms = data[:, 5].astype(int)
puissances_recues = data[:, 6]

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

# Définir la taille de la grille et le nombre de cellules
x_grid = np.arange(x_min, x_max + size_urb, size_urb)  # Ajoutez 1 à grid_size pour avoir le bon nombre de cellules
y_grid = np.arange(y_min, y_max + size_urb, size_urb)

print("longueur x_grid", len(x_grid))
print("longueur y_grid", len(y_grid))

grille = []
centres_lambert = []
sommets1_lambert = []
sommets2_lambert = []
sommets3_lambert = []
sommets4_lambert = []

# Calculer la moyenne des points dans chaque cellule de la grille
for i in range(len(x_grid) - 1):
    for j in range(len(y_grid) - 1):
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
            releves = list(zip(ids[indices], [int(e) for e in dbms[indices]], technos[indices]))
            grille.append({
                'c_l': centre_lambert,
                's1_l': sommet1_lambert,
                's2_l': sommet2_lambert,
                's3_l': sommet3_lambert,
                's4_l': sommet4_lambert,
                'releves': releves,
                'dbm_moy': float(average_power),
                'type': None,
                'area': 0})
        else:
            grille.append({
                'c_l': centre_lambert,
                's1_l': sommet1_lambert,
                's2_l': sommet2_lambert,
                's3_l': sommet3_lambert,
                's4_l': sommet4_lambert,
                'releves': [],
                'dbm_moy': 0,
                'type': None,
                'area': 0})


def converter(x, y):
    lambert93 = 'epsg:2154'  # Lambert 93
    wgs84 = 'epsg:4326'  # WGS84 (lat/lon)

    gdf_object = gpd.GeoDataFrame(None, geometry=gpd.points_from_xy(x, y), crs=lambert93)

    gdf_object = gdf_object.to_crs(wgs84)

    return gdf_object.geometry.x, gdf_object.geometry.y


x_centres_gps, y_centres_gps = converter(np.array(centres_lambert)[:, 0], np.array(centres_lambert)[:, 1])
x_sommets1_gps, y_sommets1_gps = converter(np.array(sommets1_lambert)[:, 0], np.array(sommets1_lambert)[:, 1])
x_sommets2_gps, y_sommets2_gps = converter(np.array(sommets2_lambert)[:, 0], np.array(sommets2_lambert)[:, 1])
x_sommets3_gps, y_sommets3_gps = converter(np.array(sommets3_lambert)[:, 0], np.array(sommets3_lambert)[:, 1])
x_sommets4_gps, y_sommets4_gps = converter(np.array(sommets4_lambert)[:, 0], np.array(sommets4_lambert)[:, 1])
print(len(x_centres_gps))

for i in range(len(x_centres_gps)):
    grille[i]['centre_gps'] = (float(x_centres_gps[i]), float(y_centres_gps[i]))
    grille[i]['s1_gps'] = (float(x_sommets1_gps[i]), float(y_sommets1_gps[i]))
    grille[i]['s2_gps'] = (float(x_sommets2_gps[i]), float(y_sommets2_gps[i]))
    grille[i]['s3_gps'] = (float(x_sommets3_gps[i]), float(y_sommets3_gps[i]))
    grille[i]['s4_gps'] = (float(x_sommets4_gps[i]), float(y_sommets4_gps[i]))
    grille[i]['polygon_object'] = Polygon([(float(x_sommets1_gps[i]), float(y_sommets1_gps[i])),
                                           (float(x_sommets1_gps[i]), float(y_sommets1_gps[i])),
                                           (float(x_sommets2_gps[i]), float(y_sommets2_gps[i])),
                                           (float(x_sommets3_gps[i]), float(y_sommets3_gps[i])),
                                           (float(x_sommets4_gps[i]), float(y_sommets4_gps[i]))
                                           ])

shapes_files = {
    "RUR": rurales_file,
    "PER": peri_urbaines_file,
    "URB": urbaines_file,
}

grille_polygons = [carre["polygon_object"] for carre in grille]

grille_gdf = gpd.GeoDataFrame(geometry=grille_polygons)
grille_gdf.crs = "EPSG:4326"

for key, value in shapes_files.items():

    print("Extracting file %s " % value)
    if value is not None:
        shapefile_gdf = gpd.read_file(value)
        shapefile_gdf.crs = "EPSG:4326"
        grille_gdf_reprojected = grille_gdf.to_crs(shapefile_gdf.crs)

        joined_gdf = gpd.sjoin(grille_gdf_reprojected, shapefile_gdf, how="left", predicate="intersects")

        joined_gdf = joined_gdf.dropna(subset=["index_right"])
        joined_gdf["index_right"] = joined_gdf["index_right"].astype(int)

        with alive_bar(len(joined_gdf)) as bar:
            for idx, row in joined_gdf.iterrows():
               if not pd.isna(row["index_right"]):
                shapefile_poly = shapefile_gdf.iloc[row["index_right"]]["geometry"]
                intersection_area = grille_gdf.iloc[idx]["geometry"].intersection(shapefile_poly).area
                overlap_percentage = (intersection_area / grille_gdf.iloc[idx]["geometry"].area) * 100

                if grille[idx]["area"] < overlap_percentage:
                    grille[idx]["type"] = key
                    grille[idx]["area"] = overlap_percentage
                else:
                    grille[idx]["area"] = 0
                bar()

len_x = len(x_grid)
len_y = len(y_grid)

grid = np.array(grille).reshape(len_x - 1, len_y - 1)
res = []


print(grid.shape)

def distance_gps(coord1, coord2):
    # Formule de calcul de la distance entre deux points GPS
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # Rayon de la Terre en kilomètres
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def trouver_point_proche(matrice, coord_centre):
    distances = []
    for i in range(len(matrice)):
        for j in range(len(matrice[0])):
            distance = distance_gps(coord_centre, matrice[i][j]['centre_gps'])
            distances.append((distance, (i, j)))
    return min(distances)[1]


# Trouver le point le plus proche du centre_gps
pt_1 = trouver_point_proche(grid, (1.309, 47.583))
pt_2 = trouver_point_proche(grid, (1.908, 47.889))
pt_3 = trouver_point_proche(grid, (3.575, 47.775))
print(pt_1, pt_2, pt_3)

import networkx as nx

def find_max_path(matrix, start, end):
    # Trouver le minimum de toutes les valeurs
    min_value = min(matrix[i][j]['dbm_moy'] for row in matrix for j in range(len(row)) for i in range(len(row)))
    
    # Ajouter le minimum à chaque valeur
    for row in matrix:
        for cell in row:
            if cell['dbm_moy'] != 0:
                cell['dbm_moy'] += abs(min_value)
    
    # Création d'un graphe pondéré à partir de la matrice modifiée
    G = nx.Graph()
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            if i > 0:
                weight = matrix[i][j]['dbm_moy'] + matrix[i-1][j]['dbm_moy']
                G.add_edge((i, j), (i-1, j), weight=weight)
            if j > 0:
                weight = matrix[i][j]['dbm_moy'] + matrix[i][j-1]['dbm_moy']
                G.add_edge((i, j), (i, j-1), weight=weight)
            if i < rows - 1:
                weight = matrix[i][j]['dbm_moy'] + matrix[i+1][j]['dbm_moy']
                G.add_edge((i, j), (i+1, j), weight=weight)
            if j < cols - 1:
                weight = matrix[i][j]['dbm_moy'] + matrix[i][j+1]['dbm_moy']
                G.add_edge((i, j), (i, j+1), weight=weight)
    
    # Recherche du chemin le plus court avec l'algorithme de Dijkstra
    shortest_path = nx.shortest_path(G, source=start, target=end, weight='weight')
    
    return shortest_path


path  = find_max_path(grid, pt_1, pt_3)

res = []
for node in path:
    i, j = node
    res.append({
            's1_gps': grid[i][j]['s1_gps'],
            's2_gps': grid[i][j]['s2_gps'],
            's3_gps': grid[i][j]['s3_gps'],
            's4_gps': grid[i][j]['s4_gps'],
            'dbm_moy': grid[i][j]['dbm_moy'],
            'dbm_count': len(grid[i][j]['releves']),
            'type': grid[i][j]['type']
        })
print(len(res))

polygons = []
labels_type = []
labels_moy = []

if not os.path.exists("output/"):
    os.mkdir("output")

area_urb = 0
area_per = 0
area_rur = 0


moy_values, squares = [], []

with alive_bar(len(res)) as bar:
    for entry in res:
        x1, y1 = entry["s1_gps"]
        x2, y2 = entry["s2_gps"]
        x3, y3 = entry["s3_gps"]
        x4, y4 = entry["s4_gps"]
        moy = entry["dbm_moy"]
        count = entry["dbm_count"]

        if count < nb_valeur_moy:
            continue

        polygon = sg.Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
        polygons.append(polygon)
        x = [x1, x2, x3, x4]
        y = [y1, y2, y3, y4]

        if moy != 0:
            moy_values.append(moy)
            squares.append((x,y))

        ctype = entry["type"]
        color = None
        label = None
        if ctype == "URB":
            color = "r"
            label = "Urbain"

        elif ctype == "RUR":
            color = "g"
            label = "Rural"
        elif ctype == "PER":
            color = "b"
            label = "Peri-Rural"
        else:
            color = "k"
            label = "None"

        labels_type.append(label)

        if moy >= -85:
            color = "r"
            label = "Bonne"
        elif moy < -85 and moy >= -105:
            color = "g"
            label = "Moyenne"
        elif moy < -105:
            color = "b"
            label = "Mauvaise"
        if moy == 0:
            color = "gray"
            label = "Null"
        labels_moy.append(label)

        bar()


cmap = cm.get_cmap('RdYlGn')
# Normalize the moy values to [0, 1] for color mapping
print(f"Min : {min(moy_values)} and Max : {max(moy_values)}")
norm = plt.Normalize(min(moy_values), max(moy_values))

# Plot each square with its corresponding color based on the normalized moy value
for i in range(len(moy_values)):
    moy, x , y = moy_values[i], squares[i][0], squares[i][1]
    color = cmap(norm(moy))
    plt.fill(x, y, color=color)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# waring color map
#plt.colorbar(sm, label='Moy Value')

# Show the plot
plt.gca().set_aspect('equal', adjustable='box')

gdf = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')
gdf["label"] = labels_type
gdf.to_file('output/output_type.shp')
print("Saved shapefile to output/output_type")

gdf = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')
gdf["label"] = labels_moy
gdf.to_file('output/output_moy.shp')
print("Saved shapefile to output/output_moy")

title = "Opérateur " + selected_operator + ", Techno " + selected_techno + ", Taille " + str(size_urb/1000) + " km"
plt.title(title)

print("---Execution time : %s seconds ---" % (time.time() - start_time))

plt.show()

print(area_urb, area_rur, area_per)
