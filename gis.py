import geopandas as gpd
data = gpd.read_file('population_HCMC/population_shapefile/Population_Ward_Level.shp')
data["Ratio_Increase"] = data["Pop_2019"] / data["Pop_2009"]
data["Pop_Change"] = abs(data["Pop_2019"] - data["Pop_2009"])
data["Density"] = data["Pop_2019"] / data["Shape_Area"]

max_area_com = data[data.Shape_Area == data.Shape_Area.max()]["Com_Name"].values[0]
max_area_dist = data[data.Shape_Area == data.Shape_Area.max()]["Dist_Name"].values[0]
print("Phường có diện tích lớn nhất: ", max_area_com, "thuộc quận ",max_area_dist)

max_pop_com = data[data.Pop_2019 == data.Pop_2019.max()]["Com_Name"].values[0]
max_pop_dist = data[data.Pop_2019 == data.Pop_2019.max()]["Dist_Name"].values[0]
print("Phường có dân số 2019 lớn nhất: ", max_pop_com, "thuộc quận ",max_pop_dist)

min_area_com = data[data.Shape_Area == data.Shape_Area.min()]["Com_Name"].values[0]
min_area_dist = data[data.Shape_Area == data.Shape_Area.min()]["Dist_Name"].values[0]
print("Phường có diện tích nhỏ nhất: ", min_area_com, "thuộc quận ",min_area_dist)

min_pop_com = data[data.Pop_2019 == data.Pop_2019.min()]["Com_Name"].values[0]
min_pop_dist = data[data.Pop_2019 == data.Pop_2019.min()]["Dist_Name"].values[0]
print("Phường có dân số 2019 thấp nhất: ", min_pop_com, "thuộc quận ",min_pop_dist)

max_ratio_com = data[data.Ratio_Increase == data.Ratio_Increase.max()]["Com_Name"].values[0]
max_ratio_dist = data[data.Ratio_Increase == data.Ratio_Increase.max()]["Dist_Name"].values[0]
print("Phường có tốc độ tăng trưởng dân số nhanh nhất: ", max_ratio_com, "thuộc quận ",max_ratio_dist)

min_ratio_com = data[data.Ratio_Increase == data.Ratio_Increase.min()]["Com_Name"].values[0]
min_ratio_dist = data[data.Ratio_Increase == data.Ratio_Increase.min()]["Dist_Name"].values[0]
print("Phường có tốc độ tăng trưởng dân số thấp nhất: ", min_ratio_com, "thuộc quận ",min_ratio_dist)

max_change_com = data[data.Pop_Change == data.Pop_Change.max()]["Com_Name"].values[0]
max_change_dist = data[data.Pop_Change == data.Pop_Change.max()]["Dist_Name"].values[0]
print("Phường có biến động dân số nhanh nhất: ", max_change_com, "thuộc quận ",max_change_dist)

min_change_com = data[data.Pop_Change == data.Pop_Change.min()]["Com_Name"].values[0]
min_change_dist = data[data.Pop_Change == data.Pop_Change.min()]["Dist_Name"].values[0]
print("Phường có biến động dân số chậm nhất: ", min_change_com, "thuộc quận ",min_change_dist)

max_density_com = data[data.Density == data.Density.max()]["Com_Name"].values[0]
max_density_dist = data[data.Density == data.Density.max()]["Dist_Name"].values[0]
print("Phường có mật độ dân số 2019 cao nhất: ", max_density_com, "thuộc quận ",max_density_dist)

min_density_com = data[data.Density == data.Density.min()]["Com_Name"].values[0]
min_density_dist = data[data.Density == data.Density.min()]["Dist_Name"].values[0]
print("Phường có mật độ dân số 2019 thấp nhất: ", min_density_com, "thuộc quận ",min_density_dist)





