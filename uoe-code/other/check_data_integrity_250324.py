import os 
import xarray

files = os.listdir("/home/sun/data/download_data/data/model_data/T2M/")

#print(files)
for ff in files:
    f0 = xarray.open_dataset("/home/sun/data/download_data/data/model_data/T2M/"+ff)

    print('Successfully read it !!')