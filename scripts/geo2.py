import pandas as pd
import geopandas as gpd
from functools import partial
import shapely

data = pd.read_csv('data.csv', delimiter=';')

x = data['X']
y = data['Y']

lambert93 = 'epsg:2154' # Lambert 93
wgs84 = 'epsg:4326' # WGS84 (lat/lon)

gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(x, y), crs=lambert93)

gdf = gdf.to_crs(wgs84)

data['lon'] = gdf.geometry.x
data['lat'] = gdf.geometry.y

data.to_csv('data_with_lon_lat2.csv', sep=';', index=False)
