'''
2025-7-23
This script is to monitor upward MA60 signal for enterprise stocks.
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
spot_df = spot_df[(spot_df[pe_col] > 0) & (spot_df[pe_col] < 50)]

# 筛选出来50e市值以上的
spot_df = spot_df[spot_df['总市值'] > 1e10]

# 1. Upward cross of MA60
meet_code = [] ; meet_name = []

for index, row in spot_df.iterrows():
    code = row['代码']
    name = row['名称']

    print(f"Processing {code} - {name}...")
    testa = cal_base_index(code, start_date, end_date)

    if testa is None:
        continue

    if testa['cross_last7'].iloc[-1]:
        print(f"Upward MA60 signal detected for {code} - {name}")
        meet_code.append(code)
        meet_name.append(name)

result_df = pd.DataFrame({
    '代码': meet_code,
    '名称': meet_name
})


# ====================== Send Email ======================
html_content = result_df.to_html(index=False)

import smtplib
import email.utils
from email.mime.text import MIMEText

# 编写邮件内容
message = MIMEText(html_content, 'html', 'utf-8')
message['To'] = email.utils.formataddr(('sun', 'sunweihao97@gmail.com'))
message['From'] = email.utils.formataddr(('weihao', '2309598788@qq.com'))
message['Subject'] = '股票信号提醒 MA60 上穿'

# 登录服务器并发送
server = smtplib.SMTP_SSL('smtp.qq.com', 465)
server.login('2309598788@qq.com', 'aoyqtzjhzmxaeafg')  # 替换为你的 QQ 邮箱 SMTP 授权码

server.set_debuglevel(True)

try:
    server.sendmail('2309598788@qq.com', ['sunweihao97@gmail.com'], msg=message.as_string())
finally:
    server.quit()
    

# 2. Upward cross of OBV
meet_code = [] ; meet_name = []

for index, row in spot_df.iterrows():
    code = row['代码']
    name = row['名称']

    print(f"Processing {code} - {name}...")
    testa = cal_base_index(code, start_date, end_date)

    if testa is None:
        continue

    if testa['OBV_ratio'].iloc[-1]> 0.6 and testa['OBV_slope20_pos'].iloc[-1]>0:
        print(f"OBV signal detected for {code} - {name}")
        meet_code.append(code)
        meet_name.append(name)

result_df2 = pd.DataFrame({
    '代码': meet_code,
    '名称': meet_name
})


# ====================== Send Email ======================
html_content = result_df2.to_html(index=False)


# 编写邮件内容
message = MIMEText(html_content, 'html', 'utf-8')
message['To'] = email.utils.formataddr(('sun', 'sunweihao97@gmail.com'))
message['From'] = email.utils.formataddr(('weihao', '2309598788@qq.com'))
message['Subject'] = '股票信号提醒 OBV'

# 登录服务器并发送
server = smtplib.SMTP_SSL('smtp.qq.com', 465)
server.login('2309598788@qq.com', 'aoyqtzjhzmxaeafg')  # 替换为你的 QQ 邮箱 SMTP 授权码

server.set_debuglevel(True)

try:
    server.sendmail('2309598788@qq.com', ['sunweihao97@gmail.com'], msg=message.as_string())
finally:
    server.quit()