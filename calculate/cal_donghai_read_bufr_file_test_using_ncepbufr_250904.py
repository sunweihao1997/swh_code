#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import numpy as np
import ncepbufr  # NCEPLIBS-bufr 的 Python 封装

# 你的 BUFR/PrepBUFR 文件路径（示例：GFS 的 prepbufr）
bufr_path = "gfs.2021013000.prepbufr"  # 或任意 .bufr / .bufr_d / .prepbufr 文件

# 常用的 BUFR 助记符（mnemonics）
# 头信息（站号、经纬度、相对时差、类型、海拔、平台ID等）
HDSTR = "SID XOB YOB DHR TYP ELV SAID T29"
# 观测值（按层；这里挑了常见几项）
OBSTR = "POB QOB TOB UOB VOB PRSS"
# 质量控制（可选）
QCSTR = "PQM QQM TQM WQM"

bufr = ncepbufr.open("/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr")

# 外层循环：按消息（message）
while bufr.advance() == 0:
    # 每条消息自带参考时间（YYYYMMDDHH 整点）
    ref_time = datetime.strptime(str(bufr.msg_date), "%Y%m%d%H")

    # 内层循环：按子集（subset，通常是一条站报/一条廓线）
    while bufr.load_subset() == 0:
        # 读头信息与观测
        hdr = bufr.read_subset(HDSTR).squeeze()     # shape: (n_fields,) 或 (n_fields, )
        obs = bufr.read_subset(OBSTR)               # shape: (n_fields, n_levels)
        qcf = bufr.read_subset(QCSTR)               # 同上，可选

        # 站号（可能是 IA5 字符串）；有些文件没有站号，做个保护
        try:
            # 老接口里 SID 读出类型可能是字节数组，这里统一转成字符串
            station_id = hdr[0].tostring().decode("ascii").strip()
        except Exception:
            # 某些数据类型没有 SID，用 SAID（平台/传感器ID）兜底
            station_id = f"SAID:{int(hdr[6])}" if hdr.size > 6 else "UNKNOWN"

        # 经纬度（单位通常是度）
        lon = float(hdr[1])  # XOB
        lat = float(hdr[2])  # YOB

        # 观测时间：DHR 表示相对 ref_time 的小时偏移（可能是浮点）
        obstime = ref_time + timedelta(seconds=float(hdr[3]) * 3600.0)

        # 要素数组按“要素×层数”，逐层打印一个简例：
        # 这里示范取温度 TOB、风 U/V、站压 PRSS（如果存在）
        # 注意：不同数据集并不一定都有这些要素
        names = OBSTR.split()
        def try_get(name):
            if name in names:
                idx = names.index(name)
                return np.array(obs[idx]).astype(float)
            return np.array([])

        POB  = try_get("POB")   # 压力（层）
        TOB  = try_get("TOB")   # 温度（层）
        UOB  = try_get("UOB")   # U 风（层）
        VOB  = try_get("VOB")   # V 风（层"])
        PRSS = try_get("PRSS")  # 站压（有时只有地面层）

        nlev = obs.shape[-1] if obs.ndim == 2 else 1

        print(f"{obstime:%Y-%m-%d %H:%M} {station_id:>12s} "
              f"lat={lat:7.3f} lon={lon:8.3f} levels={nlev}")

        # 举例：打印前几层的温度/风
        for k in range(min(nlev, 3)):  # 只演示前 3 层
            t = TOB[k] if TOB.size else np.nan
            u = UOB[k] if UOB.size else np.nan
            v = VOB[k] if VOB.size else np.nan
            p = POB[k] if POB.size else np.nan
            print(f"  k={k:02d}  P={p:8.1f}  T={t:8.2f}  U={u:8.2f}  V={v:8.2f}")

bufr.close()
