'''
2024-4-3
This script is to visualize the Nino34 index during the 1980-2020
'''
import xarray as xr
import numpy as np
import pandas as pd

sst_file = xr.open_dataset('/home/sun/data/process/HadISST/HadISST_SST_Nino34_month_1940_2020.nc').sel(year=slice(1980, 2020))

#print(sst_file)
# Convert into 1-D array
year     = sst_file.year.data ; month    = sst_file.month.data

nino34_1d = np.zeros((len(year) * len(month)))
print(nino34_1d.shape)
for yy in range(len(year)):
    nino34_1d[yy*12:yy*12+12] = sst_file['month_nino34'].data[yy]

nino34_1d -= np.average(nino34_1d)

#nino34_1d = np.convolve(nino34_1d,np.ones(3)/3,mode='same')

# Define the year fulfilling the El Nino/La Nina
def judge_enso(nino, year, type0):
    '''This function calculate Nov-Dec-Jan Nino34 to judge is this fulfil an El nino events'''
    Nov_nino = nino[12 * (year - 1980) + 10]
    Dec_nino = nino[12 * (year - 1980) + 11]
    Jan_nino = nino[12 * (year - 1980) + 12]
    

    if type0 == 'Nino':
        if Nov_nino > 0.5 and Dec_nino > 0.5 and Jan_nino > 0.5 and np.average(nino[12 * (year - 1980) + 10 : 12 * (year - 1980) + 13]) > np.average(nino[12 * (year - 1980) + 13 : 12 * (year - 1980) + 16]):
            return year
    elif type0 == 'Nina':
        if Nov_nino < -0.5 and Dec_nino < -0.5 and Jan_nino < -0.5 and np.average(nino[12 * (year - 1980) + 10 : 12 * (year - 1980) + 13]) < np.average(nino[12 * (year - 1980) + 13 : 12 * (year - 1980) + 16]):
            return year
    else:
        return 0

Nino_year = [] ; Nina_year = []
for yyyy in range(1980, 2020):
    Nino_year.append(judge_enso(nino34_1d, yyyy, 'Nino'))
    Nina_year.append(judge_enso(nino34_1d, yyyy, 'Nina'))

#print(Nina_year)



# Start plot
import matplotlib.pyplot as plt

time_index = pd.date_range('1980', '2021', freq='5Y')

plt.figure(figsize=(12,6))

plt.plot(np.linspace(0, len(year) * len(month), len(year) * len(month)), nino34_1d, 'k',  alpha=0.9)
plt.plot([0., len(year) * len(month)], [0.5, 0.5], 'r--', alpha=0.7)
plt.plot([0., len(year) * len(month)], [-0.5, -0.5], 'b--', alpha=0.7)

plt.xlim(0, len(year) * len(month))

plt.xticks(range(0, len(year) * len(month), 12*5), time_index.year)

# Add red/blue circle to indicate Elnino and Lanina
for nnnn in Nino_year:
    if nnnn is not None:
        plt.scatter((nnnn - 1980) * 12 + 11, nino34_1d[(nnnn - 1980) * 12 + 11], 60, facecolor='none', edgecolor='red', linewidths=2)

for nnnn in Nina_year:
    if nnnn is not None:
        plt.scatter((nnnn - 1980) * 12 + 11, nino34_1d[(nnnn - 1980) * 12 + 11], 60, facecolor='blue', edgecolor='blue', linewidths=2)

#plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20*12))
#plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(12))

plt.grid(axis='x', color='0.95')

plt.savefig('/home/sun/paint/HadISST/Nino34_1980-2020.png')