#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import numpy as np
import ncepbufr

BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"  # TODO: 改成你的文件名

# 常见/通用助记符候选清单（够你入门 + 便于“盲探”）
CANDIDATES = [
    # 头信息 & 元数据
    "SID","SAID","XOB","YOB","DHR","TYP","ELV","T29","RCYR","RCMO","RCDY","RCHR","RCMN",
    # 地面/近地层
    "PRSS","POB","TOB","QOB","UOB","VOB","ZOB","HOB","WDIR","WSPD",
    # 观测/平台相关
    "ITT","SORC","RSST","RSLP","PMSL",
    # 质量控制（如有）
    "PQM","QQM","TQM","WQM","PMQ","QMPR","QMWS",
]

def is_good(x):
    """判断一个值/数组是否包含有效数值"""
    if isinstance(x, np.ma.MaskedArray):
        arr = x.filled(np.nan)
    else:
        arr = np.array(x, dtype=float, copy=False)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return np.any(np.isfinite(arr))

def safe_str_sid(val):
    # 某些文件里 SID 是 IA5/字节数组，这里尽量解成字符串
    try:
        return val.tostring().decode("ascii", errors="ignore").strip()
    except Exception:
        try:
            return str(val)
        except Exception:
            return ""

bufr = ncepbufr.open(BUFR_FILE)
found = False

while bufr.advance() == 0 and not found:
    ref_time = datetime.strptime(str(bufr.msg_date), "%Y%m%d%H")
    while bufr.load_subset() == 0:
        print(f"# Message ref time: {ref_time}  (BUFR msg_date={bufr.msg_date})")
        good = []
        for mnem in CANDIDATES:
            try:
                arr = bufr.read_subset(mnem)
                if is_good(arr):
                    good.append(mnem)
            except Exception:
                # 该助记符在本子集/表中不存在，忽略
                pass

        print("## 这个子集里【有值】的助记符：")
        print(" ".join(good) if good else "(没探到常见字段)")

        # 演示打印几项典型值
        def get1(name):
            try:
                a = bufr.read_subset(name)
                # 对 IA5 的 SID 特殊处理
                if name == "SID":
                    return safe_str_sid(a)
                # 其他一律转 float 并取第 1 个元素（如果是二维就取 [0,0]）
                arr = np.array(a, dtype=float)
                return arr.flat[0] if arr.size > 0 else np.nan
            except Exception:
                return np.nan

        lon = get1("XOB"); lat = get1("YOB")
        dhr = get1("DHR")
        obstime = ref_time + timedelta(seconds=float(dhr)*3600.0) if np.isfinite(dhr) else ref_time
        sid = get1("SID")
        print(f"示例：SID={sid}  lon={lon:.3f}  lat={lat:.3f}  time={obstime}")

        # 如果你只想先看一条，读到这里就停
        found = True
        break

bufr.close()
if not found:
    print("没有读到任何子集（文件可能为空或表不匹配）。")
