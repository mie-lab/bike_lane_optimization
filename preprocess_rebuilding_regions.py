import os
import pandas as pd
import geopandas as gpd

greater_polygon_mapping = {
    "zrh-1": 1,
    "zrh-2": 1,
    "zrh-3": 1,
    "zrh-4": 1,
    "zrh-5": 1,
    "zrh-6": 1,
    "zrh-8": 1,
    "zrh-7": 4,
    "zrh-10": 1,
    "zrh-11": 1,
    "zrh-9": 4,
    "zrh-12": 4,
    "zrh-13": 4,
    "zrh-14": 4,
    "zrh-15": 4,
    "zrh-16": 2,
    "zrh-17": 2,
    "zrh-18": 2,
    "zrh-19": 2,
    "zrh-20": 2,
    "zrh-21": 2,
    "zrh-22": 2,
    "zrh-23": 2,
    "zrh-24": 2,
    "zrh-27": 3,
    "zrh-28": 3,
    "zrh-31": 3,
    "zrh-32": 3,
    "zrh-41": 3,
    "zrh-42": 3,
    "zrh-43": 3,
    "zrh-44": 3,
    "zrh-45": 3,
    "zrh-46": 3,
    "zrh-47": 3,
    "zrh-48": 3,
    "zrh-49": 3,
    "zrh_50": 1,
    "zrh_51": 2,
    "zrh_52": 2,
    "zrh_2055": 3,
    "zrh_2056": 1,
    "zrh_2057": 3,
    "zrh-25": 2,
    "zrh-26": 2,
    "zrh-29": 2,
    "zrh-30": 4,
    "zrh-33": 2,
    "zrh-34": 5,
    "zrh-37": 5,
    "zrh-35": 5,
    "zrh-36": 5,
    "zrh-38": 5,
    "zrh-39": 3,
    "zrh-40": 3,
    "zrh_2053": 5,
    "zrh_2054": 5,
}

data_directory = "snman/examples/data_v2"

all_regions = gpd.read_file(os.path.join(data_directory, "inputs", "rebuilding_regions", "rebuilding_regions.gpkg"))

regions_no_mainroad = all_regions[~all_regions["description"].str.contains("main")]  # .plot(cmap="viridis")
regions_no_mainroad["assignment"] = regions_no_mainroad["description"].map(greater_polygon_mapping).fillna(0)

# merge polygons by assignment
regions_for_main = regions_no_mainroad.dissolve(by="assignment", aggfunc="first")

# adapt attributes
regions_for_main.reset_index(inplace=True)
regions_for_main["hierarchies_to_include"] = "1_main_road"  # .plot(cmap="viridis")
regions_for_main["hierarchies_to_fix"] = "None"
regions_for_main["order"] = 1000 + regions_for_main["assignment"]
regions_for_main["description"] = "zrh_main_" + regions_for_main["assignment"].astype(str)
regions_for_main["keep_all_streets"] = True

regions_together = pd.concat([regions_for_main, regions_no_mainroad])

regions_together.drop("assignment", axis=1).to_file("street_network_data/zurich/rebuilding_regions.gpkg")
