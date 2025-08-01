'''
2025-7-29
This script is to monitor stock status and calculate indicators for them.
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

sys.path.append("/home/sun/swh_code/module/")
from module_stock import cal_index_for_stock, standardize_and_normalize, map_df

sys.path.append("/home/sun/swh_code/other/pybroker/")
from module_index_calculation import cal_base_index

end_date = datetime.today().strftime("%Y%m%d")
start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")

# =============== Screening Stocks ===============

spot_df = pd.read_excel("/home/sun/data/other/real_time_250710.xlsx", dtype={"代码": str})

columns = spot_df.columns.tolist()

# 自动识别市盈率列名
pe_candidates = ["市盈率-动态", "市盈率(动)", "市盈率", "市盈率_TTM", "总市值"]
pe_col = next((col for col in pe_candidates if col in columns), None)

# 把市盈率数据拉出来然后清除小于0的
spot_df = spot_df[["代码", "名称", "总市值", pe_col]].dropna()
spot_df = spot_df[(spot_df[pe_col] > 0) & (spot_df[pe_col] < 70)]

# 筛选出来100e市值以上的
spot_df = spot_df[spot_df['总市值'] > 5e10]

# 1. Upward cross of MA60
meetm60_code = [] ; meetm60_name = []
meetobv_code = [] ; meetobv_name = []
meetobvslope_code = [] ; meetobvslope_name = []
meetrsi_code = [] ; meetrsi_name = []
meetcci_code = [] ; meetcci_name = []

for index, row in spot_df.iterrows():
    code = row['代码']
    name = row['名称']

    print(f"Processing {code} - {name}...")
    testa = cal_base_index(code, start_date, end_date)

    if testa is None:
        continue

    if testa['cross_last7'].iloc[-1]:
        print(f"Upward MA60 signal detected for {code} - {name}")
        meetm60_code.append(code)
        meetm60_name.append(name)

    if testa['OBV_ratio'].iloc[-1] > 0.6:
        print(f"OBV ratio signal detected for {code} - {name}")
        meetobv_code.append(code)
        meetobv_name.append(name)

    if testa['OBV_slope20_pos'].iloc[-1] > 0.0:
        print(f"OBVslope signal detected for {code} - {name}")
        meetobvslope_code.append(code)
        meetobvslope_name.append(name)

    if testa['rsi_pos_ratio'].iloc[-1] > 0.6:
        print(f"RSI signal detected for {code} - {name}")
        meetrsi_code.append(code)
        meetrsi_name.append(name)

    if testa['CCI_slope10'].iloc[-1] > 0.0:
        print(f"CCI signal detected for {code} - {name}")
        meetcci_code.append(code)
        meetcci_name.append(name)

result_df_cross60 = pd.DataFrame({
    '代码': meetm60_code,
    '名称': meetm60_name
})

result_df_name_no60 = set(meetobv_name) & set(meetobvslope_name) & set(meetrsi_name) & set(meetcci_name)
result_df_name_add60 = set(meetobv_name) & set(meetobvslope_name) & set(meetrsi_name) & set(meetcci_name) & set(meetm60_name)

result_df_code_no60  = set(meetobv_code) & set(meetobvslope_code) & set(meetrsi_code) & set(meetcci_code)
result_df_code_add60 = set(meetobv_code) & set(meetobvslope_code) & set(meetrsi_code) & set(meetcci_code) & set(meetm60_code)

result_df_good_no60 = pd.DataFrame({
    '名称': list(result_df_name_no60),
    '代码': list(result_df_code_no60)
})

result_df_good_add60 = pd.DataFrame({
    '名称': list(result_df_name_add60),
    '代码': list(result_df_code_add60)
})

# ====================== Send Email ======================
html_content = (
    result_df_good_no60.to_html(index=False) +
    '<br><h3>======================</h3><br>' +  # HTML 中加入分隔符
    result_df_good_add60.to_html(index=False)
)

import smtplib
import email.utils
from email.mime.text import MIMEText

# 编写邮件内容
message = MIMEText(html_content, 'html', 'utf-8')
message['To'] = email.utils.formataddr(('sun', 'sunweihao97@gmail.com'))
message['From'] = email.utils.formataddr(('weihao', '2309598788@qq.com'))
message['Subject'] = '股票信号优势'

# 登录服务器并发送
server = smtplib.SMTP_SSL('smtp.qq.com', 465)
server.login('2309598788@qq.com', 'aoyqtzjhzmxaeafg')  # 替换为你的 QQ 邮箱 SMTP 授权码

server.set_debuglevel(True)

try:
    server.sendmail('2309598788@qq.com', ['sunweihao97@gmail.com'], msg=message.as_string())
finally:
    server.quit()
