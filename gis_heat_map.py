import geopandas as gpd
data = gpd.read_file('population_HCMC/population_shapefile/Population_Ward_Level.shp')
data["Density_2019"] = data["Pop_2019"] / data["Shape_Area"]
data["Density_2009"] = data["Pop_2009"] / data["Shape_Area"]
data["Density_Change"] = data["Density_2019"] / data["Density_2009"]
data_head = data.sort_values('Density_Change', ascending=False).head()
print(data_head)





