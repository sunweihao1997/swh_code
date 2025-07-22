'''
2025-7-10
This script is a test for the indicator module
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas_ta as ta
import sys

sys.path.append("/home/sun/swh_code/module/")
from module_stock import cal_index_for_stock, standardize_and_normalize


test_a = cal_index_for_stock("600900", "20200101", "20240710")

#print(test_a.tail())

test_b = standardize_and_normalize(test_a)
test_b[0].to_csv("/home/sun/data/other/test.csv", index=False)

print(test_b[0].columns)