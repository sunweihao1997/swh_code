from ncepbufr import open as bufr_open
import numpy as np

BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
TARGET_MSG = 15658
TARGET_SUBSET = 1
MISSING_BIG = 9e9

def is_missing(x):
    try:
        xf = float(x)
        return (not np.isfinite(xf)) or (abs(xf) > MISSING_BIG)
    except Exception:
        return True

def dump_arr(name, a, max_items=12):
    if a is None:
        print(f"  [{name}] -> None", flush=True)
        return
    # 结构化？
    arr = np.atleast_1d(a)
    has_names = getattr(arr, "dtype", None) is not None and arr.dtype.names
    print(f"  [{name}] type={type(a)}, structured={bool(has_names)}", flush=True)
    if has_names:
        print(f"    dtype.names={arr.dtype.names}, shape={arr.shape}", flush=True)
        for nm in arr.dtype.names:
            col = arr[nm]
            flat = np.asarray(col).reshape(-1)
            preview = ", ".join(str(v) for v in flat[:max_items])
            print(f"    - {name}.{nm} -> len={len(flat)}; head: {preview}", flush=True)
    else:
        aa = np.asarray(a)
        print(f"    shape={aa.shape}, ndim={aa.ndim}, dtype={aa.dtype}", flush=True)
        flat = aa.reshape(-1)
        preview = ", ".join(str(v) for v in flat[:max_items])
        print(f"    head: {preview}", flush=True)

def read_1d(bufr, mnem):
    """读取单个助记符，稳定返回一维 list[float|None]"""
    raw = bufr.read_subset(mnem)
    dump_arr(mnem, raw)
    if raw is None or getattr(raw, "size", 0) == 0:
        return []
    arr = np.atleast_1d(raw)
    # 结构化：极少见（单字段），但防一手
    if getattr(arr, "dtype", None) is not None and arr.dtype.names and mnem in arr.dtype.names:
        col = arr[mnem]
        flat = np.asarray(col).reshape(-1)
        return [None if is_missing(v) else float(v) for v in flat]
    # 非结构化
    aa = np.asarray(arr)
    if aa.ndim == 2:  # (n_levels, n_fields) —— 单字段通常是第二维==1
        if aa.shape[1] >= 1:
            aa = aa[:, 0]
        else:
            aa = aa.reshape(-1)
    else:
        aa = aa.reshape(-1)
    return [None if is_missing(v) else float(v) for v in aa]

def main():
    print("👉 第一次扫描：统计 message 数...", flush=True)
    bufr = bufr_open(BUFR_FILE)
    total_msgs = 0
    target_msg_nsub = None
    while bufr.advance() == 0:
        total_msgs += 1
        if total_msgs == TARGET_MSG:
            target_msg_nsub = bufr.subsets
    bufr.close()
    print(f"📊 文件中共有 message 数量：{total_msgs}", flush=True)
    print(f"📌 目标 message #{TARGET_MSG} 的 subset 数：{target_msg_nsub}", flush=True)
    if TARGET_MSG > total_msgs:
        print("❌ 目标 message 超出文件总数", flush=True)
        return
    if target_msg_nsub is not None and (TARGET_SUBSET < 1 or TARGET_SUBSET > target_msg_nsub):
        print(f"❌ subset 越界：应在 1..{target_msg_nsub}", flush=True)
        return

    print("\n👉 第二次扫描：读取目标 message/subset...", flush=True)
    bufr = bufr_open(BUFR_FILE)
    imsg = 0
    try:
        while bufr.advance() == 0:
            imsg += 1
            if imsg < TARGET_MSG:
                continue
            if imsg > TARGET_MSG:
                break

            nsub = bufr.subsets
            print(f"✅ 到达 Message #{imsg}（subsets={nsub}）", flush=True)

            isub = 0
            while bufr.load_subset() == 0:
                isub += 1
                if isub != TARGET_SUBSET:
                    continue

                # 头部经纬度（高精度）
                hdr = bufr.read_subset("CLATH CLONH")
                dump_arr("HEADER(CLATH CLONH)", hdr)
                lat = lon = None
                if hdr is not None and getattr(hdr, "size", 0) > 0:
                    hv = np.asarray(hdr, dtype=float).reshape(-1)
                    if len(hv) >= 2 and not is_missing(hv[0]) and not is_missing(hv[1]):
                        lat = float(hv[0]); lon = float(hv[1])
                print(f"📍 Header -> CLATH={lat}  CLONH={lon}", flush=True)

                print("\n🔎 分开读取单字段，避免层/列串位：", flush=True)
                prlc = read_1d(bufr, "PRLC")   # 压力（Pa）
                wdir = read_1d(bufr, "WDIR")
                wspd = read_1d(bufr, "WSPD")

                # 如果 WSPD 全缺，尝试 U/V 计算
                if not any(v is not None for v in wspd):
                    print("🧮 WSPD 缺失，尝试用 U/V 计算", flush=True)
                    uu = read_1d(bufr, "UWND")
                    vv = read_1d(bufr, "VWND")
                    n = min(len(uu), len(vv))
                    wspd = []
                    for i in range(max(len(wdir), len(prlc), n)):
                        if i < n and uu[i] is not None and vv[i] is not None:
                            wspd.append(float((uu[i]**2 + vv[i]**2) ** 0.5))
                        else:
                            wspd.append(None)

                # 层数对齐
                nlev = max(len(prlc), len(wdir), len(wspd), 1)

                # 打印对齐后的前 12 层
                print("\n🧱 Levels（对齐后，前 12 行预览）:", flush=True)
                for i in range(min(nlev, 12)):
                    P = prlc[i] if i < len(prlc) else None
                    D = wdir[i] if i < len(wdir) else None
                    S = wspd[i] if i < len(wspd) else None
                    # 友好显示
                    P2 = "--" if P is None else P
                    D2 = "--" if D is None else D
                    S2 = "--" if S is None else S
                    print(f"   L{i+1:02d}  PRLC:{P2}  WDIR:{D2}  WSPD:{S2}", flush=True)

                # 如需完整层输出/CSV，可在这里扩展
                return  # 只读这一条 subset 后退出

    finally:
        bufr.close()

if __name__ == "__main__":
    main()
