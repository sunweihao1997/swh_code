'''
2025-8-28
This script is to get the ship IDs from the Donghai ship data
'''
import pandas as pd
import os   

ship_data_path = "/mnt/f/ERA5_ship/ship_data/"

ship_list = os.listdir(ship_data_path)

ship_list_simple = []
for ship in ship_list:
    # 去掉前缀
    if ship.startswith("202506_"):
        ship = ship[len("202506_"):]
    # 去掉后缀
    if ship.endswith(".csv"):
        ship = ship[:-len(".csv")]
    ship_list_simple.append(ship)

print(ship_list_simple)

# Save to a text file
with open("/mnt/f/ERA5_ship/donghai_ship_ID_list.txt", "w") as f:
    for ship_id in ship_list_simple:
        f.write(ship_id + "\n")
    