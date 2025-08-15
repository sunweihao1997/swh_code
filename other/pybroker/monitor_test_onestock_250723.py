'''
2025-7-23
This script is to calculate the index for stock monitoring, one stock for testing.
'''
import sys
import os
import numpy as np
from datetime import datetime, timedelta

sys.path.append("/home/sun/swh_code/other/pybroker/")
from module_index_calculation import cal_base_index

end_date = datetime.today().strftime("%Y%m%d")
start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")

testa = cal_base_index("002318", start_date, end_date)

print(testa.tail(10))