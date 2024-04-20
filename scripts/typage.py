import fiona
from shapely.geometry import Polygon, Point
import json
from pyproj import Proj, transform


def convert(x, y):
    lambert93 = Proj(init='epsg:2154') # Lambert 93
    wgs84 = Proj(init='epsg:4326') # WGS84 (lat/lon)

    x, y = transform(lambert93, wgs84, x, y)

    return x, y

peri_urbaines_file = "data/Zones PERI URBAINES 41 45 89.shp"
rurales_file = "data/Zones RURALES 41 45 89.shp"
urbaines_file = "data/Zones URBAINES 41 45 89.shp"
data_file = 'grille.json'
save_file = 'type.json'

with open(data_file) as f:
    data = json.load(f)

shapes_files = {
    "PERI": peri_urbaines_file,
    "RUR": rurales_file,
    "URB": urbaines_file
}

#shapes_files = {
    #"PERI": peri_urbaines_file,
   # "RUR": rurales_file,
   # "URB": urbaines_file
#}

for key, value in shapes_files.items():
    with fiona.open(value) as shapefile:
        for record in shapefile:
            for shape in record['geometry']['coordinates']:
                try:
                    poly = Polygon(shape[0])
                    for carre in data["grille"]:
                        coords = carre["centre_gps"]
                        point = Point(coords[0], coords[1])
                        if poly.contains(point):
                            carre["type"] = key
                except:
                    print("erreur")

with open(save_file, 'w') as f:
    json.dump(data, f, indent=4)



