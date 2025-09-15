#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed: robust Table A detection for environments where bufr.msg_type is a STRING like 'NC005032'.

- Detection priority:
  1) msg_type (if attribute exists)
     - if it's a string: return as-is (e.g., 'NC005032')
     - if it's numeric: format 'NC%03d'
  2) subset (if non-empty string)
  3) mnemonics fallback: (MTYP,MSBT) / (TYP,STYP) / etc.
  4) else 'UNKNOWN'

Also extracts WDIR/WSPD + TIME/LAT/LON/PRES over a subset range.
"""

import sys, argparse
from typing import Optional, List, Dict, Tuple

# ==== Defaults (you can override by CLI) ====
DEFAULT_BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
DEFAULT_MSG = 15658
DEFAULT_SUBSET_START = 2
DEFAULT_SUBSET_END = 11
DEFAULT_MISSING_BIG = 9e9

# Candidate mnemonics (robust across templates)
LAT_CANDIDATES  = ["CLATH", "CLAT", "LAT"]
LON_CANDIDATES  = ["CLONH", "CLON", "LON"]
PRES_CANDIDATES = ["PRLC", "PRES", "PRSS", "PRL", "PRMH"]
TIME_COMPONENTS = ["YEAR", "MNTH", "DAYS", "HOUR", "MINU", "SECO"]
TIME_FALLBACKS  = ["DATTIM", "DHR"]

TABLEA_CANDIDATE_PAIRS = [("MTYP","MSBT"), ("TYP","STYP"), ("TYPI","STYP"), ("MTY","MSBT")]
TABLEA_SINGLETS = ["MTYP","MSBT","TYP","STYP","TYPI","MTY"]


def require_modules():
    try:
        import numpy as np  # noqa
        import ncepbufr as bufrmod  # noqa
    except Exception:
        print("需要 numpy 和 ncepbufr：pip install numpy ncepbufr", file=sys.stderr)
        raise


def to_masked_1d(arr):
    import numpy as np
    if arr is None:
        return None
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.ravel(np.ma.squeeze(arr))
    a = np.ravel(np.squeeze(arr))
    return np.ma.array(a, mask=np.zeros_like(a, dtype=bool))


def safe_read(b, m):
    try:
        return b.read_subset(m)
    except Exception:
        return None


def first_available(b, cands):
    for m in cands:
        arr = safe_read(b, m)
        if arr is not None:
            return m, arr
    return None, None


def assemble_time_from_components(comp_arrays):
    import numpy as np
    comps = []
    for k in TIME_COMPONENTS:
        a = comp_arrays.get(k)
        if a is None:
            return None
        comps.append(to_masked_1d(a))
    lens = [c.size for c in comps]
    if len(set(lens)) != 1:
        n = min(lens)
        comps = [c[:n] for c in comps]
    y, m, d, H, M, S = comps
    out = []
    for i in range(min(c.size for c in comps)):
        try:
            out.append(f"{int(y[i]):04d}-{int(m[i]):02d}-{int(d[i]):02d} {int(H[i]):02d}:{int(M[i]):02d}:{int(S[i]):02d}")
        except Exception:
            out.append("")
    return np.array(out, dtype=object)


def wrap_lon_180(lon):
    import numpy as np
    if lon is None:
        return None
    return ((lon + 180) % 360) - 180


def _first_int_scalar(arr):
    import numpy as np
    if arr is None:
        return None
    v = to_masked_1d(arr)
    if v.size == 0:
        return None
    if isinstance(v, np.ma.MaskedArray) and v.mask is not np.ma.nomask:
        for i in range(v.size):
            if not bool(v.mask[i]):
                try:
                    return int(round(float(v[i])))
                except Exception:
                    return None
        return None
    try:
        return int(round(float(v[0])))
    except Exception:
        return None


def detect_table_tag(b) -> str:
    """
    Fixed logic tailored to your environment:
    - If b.msg_type is a STRING (like 'NC005032'): use it directly (no int()).
    - Else if numeric-like: NC%03d.
    - Else try subset, then mnemonic fallbacks.
    """
    # 1) msg_type（属性、且优先使用字符串版）
    mt = getattr(b, "msg_type", None)
    if mt is not None:
        # 字符串直接返回
        if isinstance(mt, str):
            s = mt.strip()
            if s:
                return s
        # 有些实现是 bytes
        try:
            import numpy as np
            if isinstance(mt, (bytes, bytearray)):
                s = mt.decode("utf-8", "ignore").strip()
                if s:
                    return s
        except Exception:
            pass
        # 数值（极少见）
        try:
            return "NC" + str(int(mt)).zfill(3)
        except Exception:
            pass

    # 2) subset（字符串）
    subset_tag = getattr(b, "subset", None)
    if subset_tag:
        try:
            s = str(subset_tag).strip()
            if s:
                return s
        except Exception:
            pass

    # 3) 助记符回退
    for a, c in TABLEA_CANDIDATE_PAIRS:
        x = safe_read(b, a)
        y = safe_read(b, c)
        xi = _first_int_scalar(x)
        yi = _first_int_scalar(y)
        if xi is not None and yi is not None:
            return f"MTYP={xi:03d}, MSBT={yi:03d}"
        if xi is not None:
            return f"MTYP={xi:03d}"
        if yi is not None:
            return f"MSBT={yi:03d}"
    for a in TABLEA_SINGLETS:
        x = safe_read(b, a)
        xi = _first_int_scalar(x)
        if xi is not None:
            prefix = "MTYP" if "TYP" in a.upper() else a.upper()
            return f"{prefix}={xi:03d}"

    # 4) 实在拿不到
    return "UNKNOWN"


def read_subset_range(bufr_file, msg_idx, subset_start, subset_end, missing_big, fill_missing, lon_wrap180):
    """
    返回：(table_tag, rows)
    rows 列：TABLE_A, subset_idx, TIME, LAT, LON, PRES, WDIR, WSPD
    """
    import numpy as np
    import ncepbufr as bufrmod

    if subset_start < 1 or subset_end < subset_start:
        raise ValueError("subset 范围不合法：需要 1 <= subset_start <= subset_end")

    b = bufrmod.open(bufr_file)
    try:
        b.set_missing_value(missing_big)
    except Exception:
        try:
            b.missing_value = missing_big
        except Exception:
            pass

    # advance 到目标 message；tag 在 load_subset 之前判定
    cur = 0
    table_tag = "UNKNOWN"
    while b.advance() == 0:
        cur += 1
        if cur == msg_idx:
            table_tag = detect_table_tag(b)
            break
    else:
        b.close()
        raise IndexError(f"未找到第 {msg_idx} 条 message")

    rows: List[Dict] = []
    subset_cur = 0
    while True:
        rc = b.load_subset()
        if rc != 0:
            break
        subset_cur += 1
        if subset_cur < subset_start:
            continue
        if subset_cur > subset_end:
            break

        # 字段读取（逐个，鲁棒）
        wdir = to_masked_1d(safe_read(b, "WDIR"))
        wspd = to_masked_1d(safe_read(b, "WSPD"))
        _, lat_r = first_available(b, LAT_CANDIDATES)
        _, lon_r = first_available(b, LON_CANDIDATES)
        _, prs_r = first_available(b, PRES_CANDIDATES)
        lat = to_masked_1d(lat_r) if lat_r is not None else None
        lon = to_masked_1d(lon_r) if lon_r is not None else None
        prs = to_masked_1d(prs_r) if prs_r is not None else None

        comp = {k: safe_read(b, k) for k in TIME_COMPONENTS}
        if all(comp[k] is not None for k in TIME_COMPONENTS):
            comp_ma = {k: to_masked_1d(comp[k]) for k in TIME_COMPONENTS}
            time_str = assemble_time_from_components(comp_ma)
        else:
            _, fb = first_available(b, TIME_FALLBACKS)
            time_str = to_masked_1d(fb).astype(object) if fb is not None else None

        cols = [c for c in [wdir, wspd, lat, lon, prs, time_str] if c is not None]
        if not cols:
            continue
        n = min(c.size for c in cols)

        def filled(x):
            if x is None:
                return None
            import numpy as np
            if isinstance(x, np.ma.MaskedArray):
                return x.filled(missing_big)[:n] if fill_missing else x[:n]
            return x[:n]

        if lon_wrap180 and lon is not None:
            lon = wrap_lon_180(lon)

        wdir_n = filled(wdir)
        wspd_n = filled(wspd)
        lat_n  = filled(lat)
        lon_n  = filled(lon)
        prs_n  = filled(prs)
        tim_n  = (time_str[:n] if time_str is not None else None)

        for i in range(n):
            rows.append({
                "TABLE_A": table_tag,
                "subset_idx": subset_cur,
                "TIME": ("" if tim_n is None else tim_n[i]),
                "LAT":  ("" if lat_n is None else lat_n[i]),
                "LON":  ("" if lon_n is None else lon_n[i]),
                "PRES": ("" if prs_n is None else prs_n[i]),
                "WDIR": ("" if wdir_n is None else wdir_n[i]),
                "WSPD": ("" if wspd_n is None else wspd_n[i]),
            })

    b.close()
    return table_tag, rows


def print_preview(table_tag, rows, limit=20):
    print(f"[Table A] {table_tag}")
    print(f"总记录数：{len(rows)}")
    for r in rows[:limit]:
        print(r)


def export_csv(path, rows):
    import csv
    if not rows:
        return
    fields = ["TABLE_A", "subset_idx", "TIME", "LAT", "LON", "PRES", "WDIR", "WSPD"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[已导出] {path} （{len(rows)} 行）")


def main():
    require_modules()
    ap = argparse.ArgumentParser(description="Extract WDIR/WSPD + time/lat/lon/pressure with FIXED Table A detection.")
    ap.add_argument("-f", "--file", default=DEFAULT_BUFR_FILE)
    ap.add_argument("-m", "--msg", type=int, default=DEFAULT_MSG)
    ap.add_argument("--subset-start", type=int, default=DEFAULT_SUBSET_START)
    ap.add_argument("--subset-end", type=int, default=DEFAULT_SUBSET_END)
    ap.add_argument("--missing-big", type=float, default=DEFAULT_MISSING_BIG)
    ap.add_argument("--fill-missing", action="store_true")
    ap.add_argument("--lon-wrap180", action="store_true")
    ap.add_argument("--export-csv", default="")
    a = ap.parse_args()

    try:
        tag, rows = read_subset_range(
            bufr_file=a.file,
            msg_idx=a.msg,
            subset_start=a.subset_start,
            subset_end=a.subset_end,
            missing_big=a.missing_big,
            fill_missing=a.fill_missing,
            lon_wrap180=a.lon_wrap180,
        )
        print_preview(tag, rows)
        if a.export_csv:
            export_csv(a.export_csv, rows)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
