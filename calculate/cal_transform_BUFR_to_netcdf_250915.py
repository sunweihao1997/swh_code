'''
2025-9-15
This script is used to convert BUFR files to NetCDF format.
'''
import os
import sys
import ncepbufr
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Define the path to your BUFR file
bufr_file = '/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr'
bufr      = ncepbufr.open(bufr_file,)
imsg = 0

imsg = 0

while bufr.advance() == 0:

    imsg += 1

    tag = getattr(bufr, "subset", None)

    if not tag:

        tag = str(bufr.msg_type).zfill(3)

    nsub = 0

    while bufr.load_subset() == 0:

        nsub += 1
        # 可选：在这里 read_subset() 拿数据

        clat  = bufr.read_subset("CLAT")
        clath = bufr.read_subset("CLATH")

        rec = {"CLAT": clat, "CLATH": clath}

        print(rec)


    #print(f"Message {imsg}: {tag}, subsets = {nsub}")

bufr.close()