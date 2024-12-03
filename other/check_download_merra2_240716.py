import xarray as xr
import os

file_all = os.listdir("/home/sun/mydown/MERRA2_2m/")

for i in range(1980, 2020):
    #print(i)
    single_year_file = []
    for ff in file_all:
        if "Nx."+str(i) in ff:
            single_year_file.append(ff)

    print(f"Now it is year {i} which contains {len(single_year_file)}")

file_all.sort()
for ff in file_all[365*38:]:
    if ff[-2:] == "nc":
        print(f'Now it is reading {ff}')
        file1 = xr.open_dataset("/home/sun/mydown/MERRA2_2m/" + ff)