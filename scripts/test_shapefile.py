#!/bin/python3

import shapefile
import shapely.geometry as sg
import matplotlib.pyplot as plt
import geopandas as gpd
import pickle
from alive_progress import alive_bar

file1 = "data/Shape Depts 41 45 89.shp"
file2 = "data/Shape Blois Orleans Auxerre.shp"
file3 = "data/Zones RURALES 41 45 89.shp"
file4 = "data/Zones URBAINES 41 45 89.shp"
file5 = "data/Zones PERI URBAINES 41 45 89.shp"

files = [file1,file2,file3,file4,file5]

polygons = []
labels_type = []
labels_moy = []

#for file in files:
#    print(file)
#    r = shapefile.Reader(file)
#
#    shape_obj = r.shapeRecords()[0]
#    geom = shape_obj.shape.__geo_interface__
#    for coord in geom["coordinates"]:
#        datas = None
#        if not isinstance(coord[0], list):
#            datas = sg.Polygon(coord)
#        else:
#            datas = sg.Polygon(coord[0])
#        x, y = datas.exterior.xy
#        plt.plot(x, y)
#        polygons.append(datas)
#
import os, sys
if not os.path.exists("output/"):
    os.mkdir("output")

with open(sys.argv[1], "rb") as fichier:
    json_data = pickle.load(fichier)
    with alive_bar(len(json_data["grille"])) as bar:
        for entry in json_data["grille"]:
            x1, y1 = entry["s1_gps"]
            x2, y2 = entry["s2_gps"]
            x3, y3 = entry["s3_gps"]
            x4, y4 = entry["s4_gps"]
            moy = entry["dbm_moy"]
            
            polygon = sg.Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            polygons.append(polygon)
            x = [x1, x2, x3, x4]
            y = [y1, y2, y3, y4]

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
            
            if moy == 0:
                color = "gray"
                label = "Null"
            labels_type.append(label)
            plt.fill(x, y,color=color)

            if moy >= -85:
                color = "r" 
                label = "Bonne"
            elif moy < -85 and moy >= -105:
                color = "g" 
                label = "Moyenne"
            elif moy < -105:
                color = "b" 
                label = "Mauvaise"
            labels_moy.append(label)

            bar()

gdf = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')
gdf["label"] = labels_type
gdf.to_file('output/output_type.shp')
print("Saved shapefile to output/output_type")

gdf = gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')
gdf["label"] = labels_moy
gdf.to_file('output/output_moy.shp')
print("Saved shapefile to output/output_moy")

plt.show()
