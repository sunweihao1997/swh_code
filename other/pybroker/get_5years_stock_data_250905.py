# -*- coding: utf-8-sig -*-
"""
使用 AkShare 串行获取 A 股全市场最近 5 年日线（默认前复权）行情。
- 不使用多线程
- 每只股票之间 sleep（含随机抖动），并周期性长休
- 重试 + 断点续跑
- 每只股票单独保存为 Excel
"""

import os
import time
import json
import random
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
from tqdm import tqdm

# ========== 可配置 ==========
OUT_DIR = "/home/sun/data_n100/stocks_price/"    # 单票输出目录
ADJUST = "qfq"                  # "qfq" 前复权 | "hfq" 后复权 | "" 不复权
YEARS_BACK = 5                  # 回溯年数

MAX_RETRY = 10                  # 单票最大重试
RETRY_BASE_SLEEP = 5.0          # 重试基础等待秒（指数退避）
RETRY_JITTER = (0.0, 0.8)       # 重试随机抖动

# —— 节流参数（关键）——
SLEEP_BETWEEN = 7               # 每只股票之间的基础休眠秒
SLEEP_JITTER = (0.5, 1.5)       # 每只间隔的抖动范围（秒）
LONG_REST_EVERY = 100           # 每抓取多少只，做一次“长休”
LONG_REST_SECONDS = 300         # 长休秒数

SHUFFLE_SYMBOLS = True          # 打乱抓取顺序以分散风控
CHECKPOINT_FILE = "/home/sun/data_n100/stocks_price/_progress_check.json"  # 断点续跑记录
# ===========================

os.makedirs(OUT_DIR, exist_ok=True)

end_dt = datetime.today()
start_dt = end_dt - timedelta(days=365 * YEARS_BACK + 5)  # +5 天缓冲
START_DATE = start_dt.strftime("%Y%m%d")
END_DATE = end_dt.strftime("%Y%m%d")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8-sig") as f:
                return set(json.load(f).get("done", []))
        except Exception:
            return set()
    return set()


def save_checkpoint(done_set):
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8-sig") as f:
            json.dump({"done": sorted(list(done_set))}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def already_downloaded(code: str) -> bool:
    return os.path.exists(os.path.join(OUT_DIR, f"{code}.xlsx"))


def get_all_a_share_list():
    """
    拉取 A 股代码表（代码、名称）
    首选 ak.stock_info_a_code_name；如失败，回退到快照接口。
    """
    try:
        code_df = ak.stock_info_a_code_name()
        if "code" not in code_df.columns:
            rename_alt = {}
            for c in code_df.columns:
                if "代码" in c:
                    rename_alt[c] = "code"
                if "名称" in c:
                    rename_alt[c] = "name"
            code_df = code_df.rename(columns=rename_alt)
        code_df = code_df[["code", "name"]].dropna().drop_duplicates()
        return code_df
    except Exception:
        spot = ak.stock_zh_a_spot_em()
        rename_alt = {}
        for c in spot.columns:
            if "代码" in c:
                rename_alt[c] = "code"
            if "名称" in c:
                rename_alt[c] = "name"
        spot = spot.rename(columns=rename_alt)
        return spot[["code", "name"]].dropna().drop_duplicates()


def fetch_one(code: str, name: str):
    """
    抓取单只股票最近 5 年日线数据；成功写入 Excel，失败抛异常。
    """
    last_e = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=START_DATE,
                end_date=END_DATE,
                adjust=ADJUST
            )
            if df is None or df.empty:
                raise RuntimeError(f"{code} 无数据（可能停牌/新股/退市）")

            rename_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "振幅": "amplitude",
                "涨跌幅": "pct_chg", "涨跌额": "chg", "换手率": "turnover"
            }
            df = df.rename(columns=rename_map)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

            df["code"] = code
            df["name"] = name
            df["adjust"] = ADJUST or "none"

            # 保存为 Excel
            excel_path = os.path.join(OUT_DIR, f"{code}.xlsx")
            df.to_excel(excel_path, index=False, engine="openpyxl")

            return

        except Exception as e:
            last_e = e
            # 指数退避 + 抖动
            wait = RETRY_BASE_SLEEP * (2 ** (attempt - 1)) + random.uniform(*RETRY_JITTER)
            time.sleep(wait)
    raise RuntimeError(f"{code} 抓取失败：{repr(last_e)}")


def main():
    print(f"抓取区间：{START_DATE} ~ {END_DATE}  复权方式：{ADJUST or 'none'}")

    codes_df = get_all_a_share_list()
    codes_df["code"] = codes_df["code"].astype(str)
    codes_df = codes_df.drop_duplicates(subset=["code"]).reset_index(drop=True)

    codes = codes_df["code"].tolist()
    name_map = dict(zip(codes_df["code"], codes_df["name"]))

    done_checkpoint = load_checkpoint()
    to_fetch = [c for c in codes if c not in done_checkpoint and not already_downloaded(c)]
    if SHUFFLE_SYMBOLS:
        random.shuffle(to_fetch)

    print(f"股票总数：{len(codes)}，本次计划下载：{len(to_fetch)}")

    failures = []
    done_now = set()

    pbar = tqdm(to_fetch, desc="串行下载", unit="stock")
    for idx, code in enumerate(pbar, start=1):
        # —— 每只之间都 sleep（基础 + 抖动）——
        if idx > 1:  # 第一只不需要
            gap = SLEEP_BETWEEN + random.uniform(*SLEEP_JITTER)
            time.sleep(gap)

        # —— 周期性长休 ——（进一步降低频率）
        if LONG_REST_EVERY > 0 and idx % LONG_REST_EVERY == 0:
            time.sleep(LONG_REST_SECONDS)

        try:
            fetch_one(code, name_map.get(code, ""))
            done_now.add(code)
        except Exception as e:
            failures.append((code, repr(e)))

        # 周期性保存进度
        if idx % 25 == 0 or code in done_now:
            save_checkpoint(done_checkpoint.union(done_now))

    # 最终保存进度
    save_checkpoint(done_checkpoint.union(done_now))
    print(f"成功：{len(done_now)}  失败：{len(failures)}")
    if failures:
        fail_path = "data/_failed_codes.txt"
        os.makedirs("data", exist_ok=True)
        with open(fail_path, "w", encoding="utf-8-sig") as f:
            for code, msg in failures:
                f.write(f"{code}\t{msg}\n")
        print(f"失败清单已保存：{fail_path}")


if __name__ == "__main__":
    main()
