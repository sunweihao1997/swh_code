import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import xgboost as xgb
import numpy as np

# 加载保存的模型
model = xgb.XGBClassifier()
model.load_model('/home/sun/xgboost_model_v2.json')
print("模型已加载！")

sys.path.append("/home/sun/swh_code/module/")
from module_stock import cal_index_for_stock, standardize_and_normalize, map_df

code = "601898"
start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")
end_date = datetime.today().strftime("%Y%m%d")

# Example of calculating indicators for a specific stock

# Special 

df_a = cal_index_for_stock(code, start_date, end_date)
df_b = standardize_and_normalize(df_a)
df_c = map_df(df_b[0])

features = [
    'rsi', 'OBV_ratio', 'CCI_slope_z', 'OBV_slope_z', 'macd_diff_slope_z',
    'MA60_slope_z', 'trix_diff_slope_z', 'CCI', 'macd_diff_slope20_z', 'CCI_slope20_z', 'OBV_slope20_z'
]
df_x = df_c[features]

# 获取每个类别的概率
proba = model.predict_proba(df_x)

# 设置阈值，如果类别 2 的概率 > 60%，则预测为类别 2，否则选择概率最大的类别
threshold = 0.6
prob_2 = proba[:, 2]  # 类别 2（上涨）的概率
y_pred = np.where(prob_2 > threshold, 2, np.argmax(proba, axis=1))  # 如果类别 2 的概率大于阈值，预测为 2，否则选择概率最高的类别

# 添加预测结果到 DataFrame
df_x['prediction'] = y_pred
df_x['date'] = df_c['date']
df_x['probability'] = prob_2

# 保存预测结果
df_x.to_csv(f"/home/sun/wd_14/data/data/other/prediction_{code}.csv", index=False)

print(f"预测结果已保存为 prediction_{code}.csv")
