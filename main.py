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
import argparse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

start_time = time.time()

def converter(x, y):
    lambert93 = 'epsg:2154'  # Lambert 93
    wgs84 = 'epsg:4326'  # WGS84 (lat/lon)

    gdf_object = gpd.GeoDataFrame(None, geometry=gpd.points_from_xy(x, y), crs=lambert93)

    gdf_object = gdf_object.to_crs(wgs84)

    return gdf_object.geometry.x, gdf_object.geometry.y

def replace_with_big_square(i, j, size, value, grid, res):
    s1_gps = grid[i][j]['s1_gps']
    s2_gps = grid[i][j + size - 1]['s4_gps']
    s4_gps = grid[i + size - 1][j]['s2_gps']
    s3_gps = grid[i + size - 1][j + size - 1]['s3_gps']
    dbm_somme = 0
    dbm_count = 0
    for row in range(i, i + size):
        for col in range(j, j + size):
            if grid[row][col]['dbm_moy'] != 0:
                dbm_somme += grid[row][col]['dbm_moy']
                dbm_count += len(grid[row][col]['releves'])

    dbm_moy = dbm_somme / dbm_count if dbm_count != 0 else 0.
    res.append({
        's1_gps': s1_gps,
        's2_gps': s2_gps,
        's3_gps': s3_gps,
        's4_gps': s4_gps,
        'dbm_moy': dbm_moy,
        'dbm_count': dbm_count,
        'type': value
    })

def init():

    global filename, peri_urbaines_file, rurales_file, urbaines_file
    global size_urb, selected_techno, selected_operator, nb_valeur_moy, predict_moy

    parser = argparse.ArgumentParser(description='Challenge 3 command line tool')

    parser.add_argument('--filename', type=str, default="data/Mesures sur 41 45 89.csv", help='the name of the data file (default: "data/Mesures sur 41 45 89.csv")')
    parser.add_argument('--peri-urbaines_file', type=str, default="data/Zones PERI URBAINES 41 45 89.shp", help='the name of the peri-urban shape file (default: "data/Zones PERI URBAINES 41 45 89.shp")')
    parser.add_argument('--rurales-file', type=str, default="data/Zones RURALES 41 45 89.shp", help='the name of the rural shape file (default: "data/Zones RURALES 41 45 89.shp")')
    parser.add_argument('--urbaines-file', type=str, default="data/Zones URBAINES 41 45 89.shp", help='the name of the urban shape file (default: "data/Zones URBAINES 41 45 89.shp")')
    parser.add_argument('--size-urb', type=int, default=1500, help='the size of the urban area (default: 1500)')
    parser.add_argument('--selected_operator', type=str, default="OP3", help='the selected operator (default: "OP1")')
    parser.add_argument('--selected_techno', type=str, default="all", choices=["4G", "5G", "all"], help='the selected technology (default: "all", choices: ["4G", "5G", "all"])')
    parser.add_argument('--predict', type=bool, default=False, help='Add the prediction')
    parser.add_argument('--nb_val_moy', type=int, default=0, help='Number of values used for average calculation')

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments
    filename = args.filename
    peri_urbaines_file = args.peri_urbaines_file
    rurales_file = args.rurales_file
    urbaines_file = args.urbaines_file
    size_urb = args.size_urb
    selected_operator = args.selected_operator
    selected_techno = args.selected_techno
    nb_valeur_moy = args.nb_val_moy
    predict_moy = args.predict

def main():

    data = np.genfromtxt(filename, delimiter=';', dtype=str, skip_header=1)

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

        if os.path.exists(value):
            print("Extracting file %s " % value)
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
        else:
            print("File %s doesn't exists !" % value)

    len_x = len(x_grid)
    len_y = len(y_grid)

    grid = np.array(grille).reshape(len_x - 1, len_y - 1)
    res = []

    types = ['URB', 'PER', 'RUR']
    for i in range(0, grid.shape[0] - 4, 4):
        for j in range(0, grid.shape[1] - 4, 4):
            nb_type_rur = 0
            for row in range(i, i + 4):
                for col in range(j, j + 4):
                    nb_type_rur += 1 if grid[row][col]['type'] == types[2] else 0

            if nb_type_rur >= 5:
                replace_with_big_square(i, j, 4, types[2], grid, res)
            else:
                tab_4 = []
                tab_1 = []
                cpt = 0
                cpt_null = 0
                for row in range(i, i + 4, 2):
                    for col in range(j, j + 4, 2):
                        nb_type_2 = 0
                        nb_type_1 = 0
                        for row2 in range(row, row + 2):
                            for col2 in range(col, col + 2):
                                nb_type_2 += 1 if grid[row2][col2]['type'] == types[1] else 0
                                nb_type_1 += 1 if grid[row2][col2]['type'] == types[0] else 0
                        if nb_type_2 >= nb_type_1 and nb_type_2 >= 1:
                            tab_4.append([row, col, 2, types[1]])
                            cpt += 4
                        elif nb_type_1 > 0:
                            for row2 in range(row, row + 2):
                                for col2 in range(col, col + 2):
                                    if grid[row2][col2]['type'] != None:
                                        tab_1.append({
                                            's1_gps': grid[row2][col2]['s1_gps'],
                                            's2_gps': grid[row2][col2]['s2_gps'],
                                            's3_gps': grid[row2][col2]['s3_gps'],
                                            's4_gps': grid[row2][col2]['s4_gps'],
                                            'dbm_moy': grid[row2][col2]['dbm_moy'],
                                            'dbm_count': len(grid[row2][col2]['releves']),
                                            'type': types[0]
                                        })
                                        cpt += 1
                        else:
                            cpt_null += 1
                        
                if cpt == 16:
                    for m in range(len(tab_4)):
                        replace_with_big_square(tab_4[m][0], tab_4[m][1], tab_4[m][2], tab_4[m][3],grid,res)
                    for m in range(len(tab_1)):
                        res.append(tab_1[m])
                elif cpt_null < 4:
                    replace_with_big_square(i, j, 4, types[2],grid,res)
                                        

    print(len(res))


    # begin regression
    if predict_moy:
        X0_train_carroyage = []
        X1_train_carroyage = []
        y_train_carroyage = []

        X0_grille_carroyage = []
        X1_grille_carroyage = []

        for carre in res:
            X0_grille_carroyage.append((carre['s2_gps'][0] - carre['s1_gps'][0])/2 + carre['s1_gps'][0])
            X1_grille_carroyage.append((carre['s4_gps'][1] - carre['s1_gps'][1])/2 + carre['s1_gps'][1])
            if carre['dbm_moy'] != 0.:
                X0_train_carroyage.append((carre['s2_gps'][0] - carre['s1_gps'][0])/2 + carre['s1_gps'][0])
                X1_train_carroyage.append((carre['s4_gps'][1] - carre['s1_gps'][1])/2 + carre['s1_gps'][1])
                y_train_carroyage.append(carre['dbm_moy'])

        X_train_carroyage = np.array([X0_train_carroyage, X1_train_carroyage]).T

        # Création du modèle k-NN avec noyau gaussien
        grilleCV_carroyage = GridSearchCV(KNeighborsRegressor(), param_grid={"n_neighbors": np.arange(1, 10)}, cv=5)
        grilleCV_carroyage.fit(X_train_carroyage, y_train_carroyage)
        best_k_carroyage = grilleCV_carroyage.best_params_['n_neighbors']
        print("Meilleur k:", best_k_carroyage)
        knn_regressor_carroyage = KNeighborsRegressor(n_neighbors=best_k_carroyage)
        knn_regressor_carroyage.fit(X_train_carroyage, y_train_carroyage)

        # Predict on the mesh grid
        X_carroyage = np.array([X0_grille_carroyage, X1_grille_carroyage]).T
        print(X_carroyage)
        Z_carroyage = knn_regressor_carroyage.predict(X_carroyage)

        for carre in res:
            carre['predict_dbm_moy'] = Z_carroyage[res.index(carre)]


        # end regression


    polygons = []
    labels_type = []
    labels_moy = []

    if not os.path.exists("output/"):
        os.mkdir("output")

    area_urb = 0
    area_per = 0
    area_rur = 0

    with open("carroyage.pkl", "wb") as fichier:
        pickle.dump(res, fichier)

    moy_values, squares = [], []

    with alive_bar(len(res)) as bar:
        for entry in res:
            x1, y1 = entry["s1_gps"]
            x2, y2 = entry["s2_gps"]
            x3, y3 = entry["s3_gps"]
            x4, y4 = entry["s4_gps"]

            count = entry["dbm_count"]

            if predict_moy:
                moy = entry["predict_dbm_moy"]
            else:
                moy = entry["dbm_moy"]

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

    import matplotlib
    # cmap = cm.get_cmap('RdYlGn')
    cmap = matplotlib.colormaps["RdYlGn"]
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
    # plt.colorbar(sm, label='Moy Value')

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

    #print(area_urb, area_rur, area_per)

if __name__ == "__main__":
    try:
        init()
        main()
    except KeyboardInterrupt:
        exit(1)
