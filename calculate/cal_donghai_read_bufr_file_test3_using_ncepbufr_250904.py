from datetime import datetime
import numpy as np
import ncepbufr

LAT_CANDS = ["YOB", "CLAT", "CLATH", "LAT"]
LON_CANDS = ["XOB", "CLON", "CLONH", "LON"]
# 时间字段优先级：DHR 相对时间 → 绝对时间 YEAR/MNTH/DAYS/HOUR/MINU/SECO → 接收时间 RCYR/RCMO/RCDY/RCHR/RCMN
TIME_GROUPS = [
    ("DHR",),  # 相对消息时间（小时）
    ("YEAR","MNTH","DAYS","HOUR","MINU","SECO"),
    ("RCYR","RCMO","RCDY","RCHR","RCMN"),  # 你现在就看到了这些
]

def read_arr(bufr, mnem):
    try:
        a = bufr.read_subset(mnem)
        if isinstance(a, np.ma.MaskedArray):
            a = a.filled(np.nan)
        return np.array(a, dtype=float, copy=False)
    except Exception:
        return np.array([])

def first_available(bufr, cand_list):
    """在候选助记符里找第一个有值的，返回 (name, array)"""
    for m in cand_list:
        arr = read_arr(bufr, m)
        if arr.size and np.isfinite(np.nanmax(arr)):
            return m, arr
    return None, np.array([])

def build_time(bufr, ref_time):
    # 1) DHR（相对消息时间，小时）
    dhr = read_arr(bufr, "DHR")
    if dhr.size and np.isfinite(dhr.flat[0]):
        return ref_time + (np.timedelta64(int(round(dhr.flat[0]*3600)), "s"))

    # 2) 绝对时间（优先 YEAR/MNTH...）
    for group in TIME_GROUPS[1:]:
        vals = [read_arr(bufr, g) for g in group]
        if all(v.size and np.isfinite(v.flat[0]) for v in vals):
            y = int(vals[0].flat[0]); mo = int(vals[1].flat[0]); dy = int(vals[2].flat[0])
            hh = int(vals[3].flat[0]); mm = int(vals[4].flat[0])
            ss = int(vals[5].flat[0]) if len(group) >= 6 else 0
            try:
                return np.datetime64(f"{y:04d}-{mo:02d}-{dy:02d}T{hh:02d}:{mm:02d}:{ss:02d}")
            except Exception:
                pass
    # 如果都没有，就返回消息参考时间
    return np.datetime64(ref_time.strftime("%Y-%m-%dT%H:%M:%S"))

def to_uv_from_dirspd(wdir_deg, wspd):
    """风向（来向，度，气象学）+ 风速(m/s) -> U/V 分量"""
    rad = np.deg2rad(wdir_deg)
    U = -wspd * np.sin(rad)
    V = -wspd * np.cos(rad)
    return U, V

# === 主流程示意 ===
bufr = ncepbufr.open("/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr")  # TODO: 替换你的文件
while bufr.advance() == 0:
    ref_time = datetime.strptime(str(bufr.msg_date), "%Y%m%d%H")
    while bufr.load_subset() == 0:
        # 经度/纬度：在候选里找
        lat_name, LAT = first_available(bufr, LAT_CANDS)
        lon_name, LON = first_available(bufr, LON_CANDS)

        # 时间：DHR 或 YEAR.. 或 RCYR.. 组装
        obstime = build_time(bufr, ref_time)

        # 风向风速
        WDIR = read_arr(bufr, "WDIR")
        WSPD = read_arr(bufr, "WSPD")

        # 只演示取第一层/单值（很多字段按层；需要可改成逐层循环）
        lat = LAT.flat[0] if LAT.size else np.nan
        lon = LON.flat[0] if LON.size else np.nan
        wdir = WDIR.flat[0] if WDIR.size else np.nan
        wspd = WSPD.flat[0] if WSPD.size else np.nan

        # 可选：把风向风速转为 U/V
        if np.isfinite(wdir) and np.isfinite(wspd):
            u, v = to_uv_from_dirspd(wdir, wspd)
        else:
            u = v = np.nan

        print(f"time={obstime} lat({lat_name})={lat:.3f} lon({lon_name})={lon:.3f} "
              f"WDIR={wdir:.1f} WSPD={wspd:.2f}  U={u:.2f} V={v:.2f}")
bufr.close()
