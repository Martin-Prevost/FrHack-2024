#!/bin/python3

import matplotlib.pyplot as plt
import geopandas

file1 = "data/Shape Depts 41 45 89.shp"
file2 = "data/Shape Blois Orleans Auxerre.shp"
file3 = "data/Zones RURALES 41 45 89.shp"
file4 = "data/Zones URBAINES 41 45 89.shp"
file5 = "data/Zones PERI URBAINES 41 45 89.shp"

files = [file1,file2,file3,file4,file5]

polygons = []
labels_type = []
labels_moy = []

for file in files:
    print(file)
    gdf = geopandas.read_file(file)
    gdf.plot(color='black')

plt.show()
