'''
2025-7-22
This script is to test the calculation module
'''
import sys
import os
import numpy as np
from datetime import datetime, timedelta

sys.path.append("/home/sun/swh_code/other/pybroker/")
from module_index_calculation import cal_base_index

end_date = datetime.today().strftime("%Y%m%d")
start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")

testa = cal_base_index("603115", start_date, end_date)

print(testa.columns)