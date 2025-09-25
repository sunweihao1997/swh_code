#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged Stock Analysis and Trading Signal Script (Standalone, Runnable)
-------------------------------------------------------------------
功能：
1) A股快照筛选（市值、PE），计算最近52周收盘价“百分位”（默认用分布百分位）。
2) 对入围股票计算技术指标（MA60、OBV、RSI、CCI），根据多重信号筛选候选股。
3) 定时执行并邮件通知（支持环境变量；无凭证则跳过邮件，仅记录日志）。

依赖：
- akshare
- pandas, numpy
- scikit-learn（仅用于线性回归斜率）
- schedule

安装：
    pip install akshare pandas numpy scikit-learn schedule

环境变量（推荐）：
    export QQ_SMTP_PASSWORD="你的QQ邮箱SMTP授权码"

注意：
- 本脚本自带技术指标计算（无需本地自定义模块）。
- 邮件默认使用 QQ 邮箱 SMTP（可替换为企业邮箱）。
- schedule 使用系统时区。若你的服务器非中国时区，请自行调整。
"""

import os
import sys
import time
import random
import smtplib
import email.utils
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import schedule
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import akshare as ak

# =============================
# 1. 配置
# =============================
LOG_FILE = os.getenv("STOCK_LOG_FILE", "/home/sun/stock_data/log/stock_log.txt")
REALTIME_DATA_DIR = os.getenv("STOCK_REALTIME_DIR", "/home/sun/stock_data/all_stock_realtime/")
PERCENTILE_DATA_DIR = os.getenv("STOCK_PERCENTILE_DIR", "/home/sun/stock_data/stock_percentile/")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "2309598788@qq.com")
EMAIL_PASSWORD = os.getenv("QQ_SMTP_PASSWORD")  # 推荐从环境变量读取
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "sunweihao97@gmail.com")

# 策略参数
MARKET_CAP_MIN = float(os.getenv("MARKET_CAP_MIN", 5e9))   # 最小总市值：50亿元
PE_RATIO_MAX = float(os.getenv("PE_RATIO_MAX", 100))      # 最大PE
MIN_DATA_POINTS = int(os.getenv("MIN_DATA_POINTS", 180))  # 过去一年最少交易日
PERCENTILE_MAX = float(os.getenv("PERCENTILE_MAX", 61.8)) # 52周百分位阈值

# 技术指标参数
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
CCI_PERIOD = int(os.getenv("CCI_PERIOD", 20))
MA60_WINDOW = int(os.getenv("MA60_WINDOW", 60))
POS_RATIO_WINDOW = int(os.getenv("POS_RATIO_WINDOW", 30))
SLOPE_WINDOW = int(os.getenv("SLOPE_WINDOW", 5))

# 目录准备
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(REALTIME_DATA_DIR, exist_ok=True)
os.makedirs(PERCENTILE_DATA_DIR, exist_ok=True)


def log_message(message: str, f):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    print(line.strip())
    f.write(line)
    f.flush()


# =============================
# 2. 工具函数：技术指标
# =============================

def calc_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    # Wilder's RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method='bfill')


def calc_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df['最高'] + df['最低'] + df['收盘']) / 3.0
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    cci = (tp - ma) / (0.015 * md)
    return cci


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    # 成交量单位：手/股 —— 直接使用相对变化即可
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume.fillna(0)).cumsum()
    return obv


def linreg_slope(y: pd.Series) -> float:
    # 返回简单线性回归斜率（x 为 0..n-1）
    y = y.dropna()
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y.values)
    return float(model.coef_[0])


# =============================
# 3. 阶段1：A股筛选 + 52周百分位
# =============================

def pick_col(df: pd.DataFrame, candidates):
    return next((c for c in candidates if c in df.columns), None)


def calculate_52_week_percentile(log_file_handle):
    log_message("--- Stage 1: 开始 A股筛选与52周百分位计算 ---", log_file_handle)
    current_date = datetime.now().strftime("%Y%m%d")
    end_date = current_date
    start_date = (datetime.today() - timedelta(days=365)).strftime("%Y%m%d")

    # 1) 拉取快照
    try:
        log_message("拉取A股实时快照...", log_file_handle)
        spot_df = ak.stock_zh_a_spot_em()
        realtime_file_path = os.path.join(REALTIME_DATA_DIR, f"real_time_{current_date}.xlsx")
        spot_df.to_excel(realtime_file_path, index=False)
        log_message(f"实时快照已保存，共 {len(spot_df)} 只股票。", log_file_handle)
    except Exception as e:
        log_message(f"ERROR: 获取/保存快照失败：{e}", log_file_handle)
        return None

    # 2) 读取并做列名映射 + 基本面筛选
    spot_df = pd.read_excel(realtime_file_path, dtype={"代码": str})

    pe_candidates = ["市盈率-动态", "市盈率TTM", "市盈率(动)", "市盈率", "滚动市盈率"]
    pe_col = pick_col(spot_df, pe_candidates)

    col_map_options = {
        "代码": ["代码", "股票代码"],
        "名称": ["名称", "股票简称"],
        "总市值": ["总市值", "总市值(元)", "总市值-按最新收盘价"],
    }

    code_col = pick_col(spot_df, col_map_options["代码"]) 
    name_col = pick_col(spot_df, col_map_options["名称"]) 
    mcap_col = pick_col(spot_df, col_map_options["总市值"]) 

    if not all([code_col, name_col, mcap_col, pe_col]):
        log_message("ERROR: 必要列缺失（代码/名称/总市值/市盈率）", log_file_handle)
        return None

    spot_df = spot_df[[code_col, name_col, mcap_col, pe_col]].dropna()
    spot_df.rename(columns={code_col: "代码", name_col: "名称", mcap_col: "总市值"}, inplace=True)

    initial_count = len(spot_df)
    spot_df = spot_df[(spot_df[pe_col] > 0) & (spot_df[pe_col] < PE_RATIO_MAX)]
    spot_df = spot_df[spot_df['总市值'] > MARKET_CAP_MIN]
    log_message(
        f"基本面筛选：{initial_count} → {len(spot_df)}（PE<{PE_RATIO_MAX} 且 总市值>{MARKET_CAP_MIN/1e8:.0f}亿）",
        log_file_handle,
    )

    # 3) 计算 52 周“分布百分位”（last_close 在样本分布中的百分位）
    def percentile_of_last(close: pd.Series) -> float:
        last = close.iloc[-1]
        n = len(close)
        if n == 0:
            return np.nan
        return 100.0 * (close.le(last).sum()) / n

    percentiles = []
    total = len(spot_df)
    for i, (_, row) in enumerate(spot_df.iterrows(), 1):
        code, name = row['代码'], row['名称']
        try:
            df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
            if df is None or df.empty or len(df) < MIN_DATA_POINTS:
                log_message(f"({i}/{total}) 跳过 {code}({name})：交易日不足 {len(df) if df is not None else 0}", log_file_handle)
                percentiles.append(np.nan)
                continue
            p = percentile_of_last(df['收盘'])
            percentiles.append(p)
            log_message(f"({i}/{total}) {code}({name}) - 52周百分位：{p:.2f}%", log_file_handle)
            time.sleep(random.randint(1, 3))  # 避免频率过高
        except Exception as e:
            log_message(f"ERROR {code}({name}): {e}", log_file_handle)
            percentiles.append(np.nan)

    spot_df['52周百分位'] = percentiles
    spot_df = spot_df.dropna(subset=['52周百分位'])

    percentile_file_path = os.path.join(PERCENTILE_DATA_DIR, f"52week_percentile_output_{current_date}.xlsx")
    spot_df.to_excel(percentile_file_path, index=False)
    log_message(f"--- Stage 1 完成：保存 {len(spot_df)} 条至 {percentile_file_path} ---", log_file_handle)
    return percentile_file_path


# =============================
# 4. 阶段2：技术指标 + 通知
# =============================

def calc_technicals(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values('日期').reset_index(drop=True)

    # MA60 及穿越信号（近7日是否发生上穿）
    df['MA60'] = calc_ma(df['收盘'], MA60_WINDOW)
    df['close_prev'] = df['收盘'].shift(1)
    df['MA60_prev'] = df['MA60'].shift(1)
    df['up_cross'] = (df['close_prev'] <= df['MA60_prev']) & (df['收盘'] > df['MA60'])

    # OBV 及派生
    df['OBV'] = calc_obv(df['收盘'], df['成交量'])
    df['OBV_MA10'] = df['OBV'].rolling(10).mean()

    # RSI/CCI
    df['RSI'] = calc_rsi(df['收盘'], RSI_PERIOD)
    df['CCI'] = calc_cci(df, CCI_PERIOD)

    # 比例与斜率
    tail = df.tail(POS_RATIO_WINDOW)
    df.loc[:, 'obv_pos_ratio'] = np.nan
    df.loc[:, 'rsi_pos_ratio'] = np.nan

    if len(tail) > 0:
        obv_pos_ratio = float((tail['OBV'] > tail['OBV_MA10']).mean())  # 最近N日中，OBV>OBV_MA10的比例
        rsi_pos_ratio = float((tail['RSI'] > 50).mean())
    else:
        obv_pos_ratio, rsi_pos_ratio = np.nan, np.nan

    cci_slope5 = linreg_slope(df['CCI'].tail(SLOPE_WINDOW))
    obv_slope5 = linreg_slope(df['OBV'].tail(SLOPE_WINDOW))

    # 收敛到最后一行作为“当前信号”输出（同时保留完整df用于调试）
    df.loc[:, 'cross_last7'] = df['up_cross'].rolling(7).max().astype(bool)
    df.loc[:, 'obv_pos_ratio_val'] = obv_pos_ratio
    df.loc[:, 'rsi_pos_ratio_val'] = rsi_pos_ratio
    df.loc[:, 'CCI_slope5_val'] = cci_slope5
    df.loc[:, 'obv_slope5_val'] = obv_slope5

    return df


def monitor_and_notify(percentile_file_path, log_file_handle):
    if not percentile_file_path or not os.path.exists(percentile_file_path):
        log_message("ERROR: 未找到阶段1结果文件，终止阶段2。", log_file_handle)
        return

    log_message("--- Stage 2: 技术指标筛选与通知 ---", log_file_handle)
    end_date = datetime.today().strftime("%Y%m%d")
    start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")

    # 1) 载入并按百分位过滤
    spot_df = pd.read_excel(percentile_file_path, dtype={"代码": str})
    spot_df = spot_df[spot_df['52周百分位'] < PERCENTILE_MAX].reset_index(drop=True)
    log_message(f"读取 {len(spot_df)} 只股票（52周百分位<{PERCENTILE_MAX}%）", log_file_handle)

    # 2) 计算技术指标并打分
    signal_lists = {'MA60': [], 'OBV_Ratio': [], 'OBV_Slope': [], 'RSI_Ratio': [], 'CCI_Slope': []}

    total = len(spot_df)
    for i, (_, row) in enumerate(spot_df.iterrows(), 1):
        code, name = str(row['代码']), str(row['名称'])
        stock_info = {'code': code, 'name': name}
        log_message(f"({i}/{total}) 计算技术指标：{code}-{name}...", log_file_handle)
        try:
            tech_df = calc_technicals(code, start_date, end_date)
            if tech_df.empty:
                continue
            last = tech_df.iloc[-1]

            if bool(last.get('cross_last7', False)):
                signal_lists['MA60'].append(stock_info)
            if last.get('obv_pos_ratio_val', 0.0) > 0.6:
                signal_lists['OBV_Ratio'].append(stock_info)
            if last.get('obv_slope5_val', 0.0) > 0.0:
                signal_lists['OBV_Slope'].append(stock_info)
            if last.get('rsi_pos_ratio_val', 0.0) > 0.6:
                signal_lists['RSI_Ratio'].append(stock_info)
            if last.get('CCI_slope5_val', 0.0) > 0.0:
                signal_lists['CCI_Slope'].append(stock_info)

            time.sleep(1.2)  # 温和节流
        except Exception as e:
            log_message(f"ERROR 技术面 {code}-{name}: {e}", log_file_handle)

    # 3) 合并信号（四项基础 + MA60 上穿为更强）
    name_sets = {k: {i['name'] for i in v} for k, v in signal_lists.items()}
    code_map = {i['name']: i['code'] for v in signal_lists.values() for i in v}

    base_criteria_names = (
        name_sets['OBV_Ratio'] &
        name_sets['OBV_Slope'] &
        name_sets['RSI_Ratio'] &
        name_sets['CCI_Slope']
    )
    all_criteria_names = base_criteria_names & name_sets['MA60']

    result_df_good_no60 = pd.DataFrame({'名称': sorted(list(base_criteria_names))})
    result_df_good_no60['代码'] = result_df_good_no60['名称'].map(code_map)

    result_df_good_add60 = pd.DataFrame({'名称': sorted(list(all_criteria_names))})
    result_df_good_add60['代码'] = result_df_good_add60['名称'].map(code_map)

    log_message(f"基础条件通过：{len(result_df_good_no60)} 只", log_file_handle)
    log_message(f"基础+MA60穿越：{len(result_df_good_add60)} 只", log_file_handle)

    # 4) 发送邮件（若配置完整）
    if result_df_good_no60.empty and result_df_good_add60.empty:
        log_message("无满足条件股票，跳过邮件发送。", log_file_handle)
        return

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 66%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h2>Stock Signals for {datetime.now().strftime('%Y-%m-%d')}</h2>
        <h3>基础条件（OBV, RSI, CCI）</h3>
        {result_df_good_no60.to_html(index=False)}
        <br><hr><br>
        <h3>基础 + MA60 上穿（较强）</h3>
        {result_df_good_add60.to_html(index=False)}
    </body>
    </html>
    """

    if EMAIL_SENDER and EMAIL_RECIPIENT and EMAIL_PASSWORD:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Stock Signals Detected - {datetime.now().strftime('%Y-%m-%d')}"
            msg['From'] = email.utils.formataddr(('Stock Monitor Bot', EMAIL_SENDER))
            msg['To'] = email.utils.formataddr(('Recipient', EMAIL_RECIPIENT))
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))

            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, [EMAIL_RECIPIENT], msg.as_string())
            log_message("邮件发送成功。", log_file_handle)
        except Exception as e:
            log_message(f"ERROR: 邮件发送失败：{e}", log_file_handle)
        finally:
            try:
                server.quit()
            except Exception:
                pass
    else:
        log_message("未配置邮箱凭证（或接收人），已跳过邮件发送。", log_file_handle)


# =============================
# 5. 主流程 & 调度
# =============================

def run_strategy_workflow():
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message(f"\n=== [{start_ts}] 开始执行工作流 ===", f)

        percentile_file = calculate_52_week_percentile(f)
        if percentile_file:
            monitor_and_notify(percentile_file, f)
        else:
            log_message("阶段1失败，终止阶段2。", f)

        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message(f"=== [{end_ts}] 工作流结束 ===\n", f)


if __name__ == "__main__":
    # 每天 09:15 运行一次（使用系统时区）
    schedule.every().day.at("09:15").do(run_strategy_workflow)

    print("Scheduler 已启动（Ctrl+C 退出）...")
    print("启动时先执行一次全流程...")
    run_strategy_workflow()
    print("已完成首次执行，等待定时任务...")

    while True:
        schedule.run_pending()
        time.sleep(30)
