#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read WDIR & WSPD from a message's subset range (inclusive) using ncepbufr,
flatten each subset's values to 1D, then concatenate across subsets
-- producing one long vector for WDIR and one for WSPD.

Defaults:
  file   : /home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr
  msg    : 15658           (1-based)
  subset-start : 2         (inclusive, 1-based)
  subset-end   : 11        (inclusive, 1-based)  # 连续10条 subset
  missing-big  : 9e9       (used with --fill-missing)

Usage:
  python read_bufr_concat_wdir_wspd_10subsets.py
  python read_bufr_concat_wdir_wspd_10subsets.py -f /path/to/file.bufr -m 15658 --subset-start 2 --subset-end 11
  python read_bufr_concat_wdir_wspd_10subsets.py --fill-missing
  python read_bufr_concat_wdir_wspd_10subsets.py --export-csv out.csv
"""

import sys
import argparse
from typing import Tuple, List, Optional

DEFAULT_BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
DEFAULT_MSG = 15658           # 1-based
DEFAULT_SUBSET_START = 2      # 1-based
DEFAULT_SUBSET_END = 11       # 1-based, inclusive; 2..11 共10条
DEFAULT_MISSING_BIG = 9e9


def require_modules():
    try:
        import numpy as np  # noqa
        import ncepbufr as bufrmod  # noqa
    except Exception:
        print("需要 numpy 和 ncepbufr：pip install numpy ncepbufr", file=sys.stderr)
        raise


def to_masked_1d(arr) -> "np.ma.MaskedArray":
    """Squeeze -> ravel to 1D masked array (even if input was ndarray)."""
    import numpy as np
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.ravel(np.ma.squeeze(arr))
    else:
        a = np.ravel(np.squeeze(arr))
        return np.ma.array(a, mask=np.zeros_like(a, dtype=bool))


def read_subset_range_concat(bufr_file: str, msg_idx: int,
                             subset_start: int, subset_end: int,
                             missing_big: float, fill_missing: bool
                             ) -> Tuple["np.ma.MaskedArray", "np.ma.MaskedArray"]:
    """
    核心流程：
      1) advance 到目标 message
      2) 在该 message 内从 subset=1 开始逐个 load_subset()
      3) 对于 subset_start..subset_end 区间内的每个 subset：
           分别 read_subset("WDIR") 与 read_subset("WSPD")
           -> squeeze+ravel 成 1D -> 追加到各自列表
      4) 最后对 WDIR/WSPD 列表分别做 np.ma.concatenate 得到长向量
    """
    import numpy as np
    import ncepbufr as bufrmod

    if subset_start < 1 or subset_end < subset_start:
        raise ValueError("subset 范围不合法：需要 1 <= subset_start <= subset_end")

    b = bufrmod.open(bufr_file)

    # （可选）设置库的缺测值；不同版本可能无效，不影响 masked 逻辑
    try:
        b.set_missing_value(missing_big)
    except Exception:
        try:
            b.missing_value = missing_big
        except Exception:
            pass

    # 1) 前进到目标 message
    msg_cur = 0
    found_msg = False
    while b.advance() == 0:
        msg_cur += 1
        if msg_cur == msg_idx:
            found_msg = True
            break
    if not found_msg:
        b.close()
        raise IndexError(f"文件中未找到第 {msg_idx} 条 message。")

    # 2) 在该 message 内逐个 subset 前进，并在范围内读取
    wdir_chunks: List["np.ma.MaskedArray"] = []
    wspd_chunks: List["np.ma.MaskedArray"] = []

    subset_cur = 0
    while True:
        rc = b.load_subset()
        if rc != 0:
            break  # 没有更多 subset
        subset_cur += 1

        if subset_cur < subset_start:
            continue
        if subset_cur > subset_end:
            break

        # 3) 单独读取，避免某些实现对 "WDIR WSPD" 的返回不一致
        def safe_read(mnem: str) -> Optional["np.ma.MaskedArray"]:
            try:
                arr = b.read_subset(mnem)
                return to_masked_1d(arr)
            except Exception:
                return None

        wdir_arr = safe_read("WDIR")
        wspd_arr = safe_read("WSPD")

        if wdir_arr is not None:
            wdir_chunks.append(wdir_arr)
        else:
            print(f"[提示] subset {subset_cur} 中无 WDIR。")

        if wspd_arr is not None:
            wspd_chunks.append(wspd_arr)
        else:
            print(f"[提示] subset {subset_cur} 中无 WSPD。")

    b.close()

    if len(wdir_chunks) == 0:
        raise RuntimeError("在指定 subset 范围内未读取到任何 WDIR 数据。")
    if len(wspd_chunks) == 0:
        raise RuntimeError("在指定 subset 范围内未读取到任何 WSPD 数据。")

    # 4) 分别拼接
    wdir_all = wdir_chunks[0]
    for ch in wdir_chunks[1:]:
        wdir_all = np.ma.concatenate([wdir_all, ch])

    wspd_all = wspd_chunks[0]
    for ch in wspd_chunks[1:]:
        wspd_all = np.ma.concatenate([wspd_all, ch])

    # 可选：把缺测 mask 填成大数
    if fill_missing:
        wdir_all = wdir_all.filled(missing_big)
        wspd_all = wspd_all.filled(missing_big)

    return wdir_all, wspd_all


def preview_vector(name: str, vec, missing_big: float, fill_missing: bool):
    import numpy as np
    print(f"=== 合并后 {name} ===")
    print("类型:", type(vec))
    is_masked = isinstance(vec, np.ma.MaskedArray)
    print("是否为 masked array：", bool(is_masked))
    print("shape：", getattr(vec, "shape", None))
    print("dtype：", getattr(vec, "dtype", None))
    flat = vec.ravel()
    nshow = min(10, flat.size)
    print(f"前 {nshow} 个值：", flat[:nshow])
    if is_masked:
        miss = int(vec.mask.sum()) if vec.mask is not np.ma.nomask else 0
        print("缺测个数：", miss)


def maybe_export_csv(path: str, wdir_vec, wspd_vec):
    """仅为方便；将两列按最小共同长度对齐导出（不同 subset 内长度可能不同，导出不保证逐条物理配对）。"""
    import numpy as np
    import csv
    a = wdir_vec.ravel()
    b = wspd_vec.ravel()
    n = min(a.size, b.size)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["WDIR", "WSPD"])
        for i in range(n):
            wr.writerow([a[i], b[i]])
    print(f"[已导出] {path}  （按最小共同长度 {n} 对齐）")


def main():
    require_modules()
    p = argparse.ArgumentParser(description="Concat WDIR & WSPD over a subset range in one BUFR message (ncepbufr).")
    p.add_argument("-f", "--file", default=DEFAULT_BUFR_FILE, help="Path to BUFR file")
    p.add_argument("-m", "--msg", type=int, default=DEFAULT_MSG, help="1-based message index")
    p.add_argument("--subset-start", type=int, default=DEFAULT_SUBSET_START, help="subset start (inclusive, 1-based)")
    p.add_argument("--subset-end", type=int, default=DEFAULT_SUBSET_END, help="subset end (inclusive, 1-based)")
    p.add_argument("--missing-big", type=float, default=DEFAULT_MISSING_BIG, help="Value when --fill-missing is used")
    p.add_argument("--fill-missing", action="store_true", help="Fill masked values to --missing-big")
    p.add_argument("--export-csv", default="", help="Optional CSV path to export [WDIR, WSPD] preview")
    args = p.parse_args()

    try:
        wdir_all, wspd_all = read_subset_range_concat(
            bufr_file=args.file,
            msg_idx=args.msg,
            subset_start=args.subset_start,
            subset_end=args.subset_end,
            missing_big=args.missing_big,
            fill_missing=args.fill_missing,
        )
        preview_vector("WDIR", wdir_all, args.missing_big, args.fill_missing)
        preview_vector("WSPD", wspd_all, args.missing_big, args.fill_missing)

        if args.export_csv:
            maybe_export_csv(args.export_csv, wdir_all, wspd_all)

    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
