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
    x,y = pickle.load(fichier)
    plt.scatter(x, y, color='blue')


points_shapely = []
for i in range(len(x)):
    points_shapely.append(sg.Point(x[i],y[i]))
gdf_points = gpd.GeoDataFrame(geometry=points_shapely, crs='EPSG:4326')

# Add any additional attributes as columns
gdf_points["attribute"] = "example_attribute"

# Save the GeoDataFrame to a shapefile
gdf_points.to_file('output/output_points.shp')
print("Saved shapefile to output/output_points.shp")

plt.show()
