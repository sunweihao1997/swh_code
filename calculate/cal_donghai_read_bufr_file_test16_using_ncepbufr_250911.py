#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust reader for GDAS BUFR via ncepbufr:
- Seeks a specific message/subset (1-based)
- Reads WDIR and WSPD individually (no reliance on tuple return)
- Prints type/shape/dtype/preview/missing counts
- Reports clearly if a mnemonic is not present in the subset

Usage:
  python read_bufr_wdir_wspd_safe.py
  python read_bufr_wdir_wspd_safe.py -f /path/to/file.bufr -m 15658 -s 1
  python read_bufr_wdir_wspd_safe.py --fill-missing --missing-big 9e9
"""

import sys
import argparse

DEFAULT_BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
DEFAULT_MSG = 15658       # 1-based
DEFAULT_SUBSET = 1        # 1-based
DEFAULT_MISSING_BIG = 9e9

MNEMONICS = ["WDIR", "WSPD"]  # 需要的量（可自行加 "UWND", "VWND" 等）


def require_modules():
    try:
        import numpy as np  # noqa
        import ncepbufr as bufrmod  # noqa
    except Exception:
        print("需要 numpy 和 ncepbufr：pip install numpy ncepbufr", file=sys.stderr)
        raise


def print_info(name, arr, fill_missing, missing_big):
    import numpy as np
    print(f"=== {name} ===")
    if arr is None:
        print("该 subset 中不存在该变量或无法读取。")
        return
    print("类型:", type(arr))
    is_masked = isinstance(arr, np.ma.MaskedArray)
    print("是否为 masked array：", bool(is_masked))
    print("shape：", getattr(arr, "shape", None))
    print("dtype：", getattr(arr, "dtype", None))
    # 可选：填充缺测
    if is_masked and fill_missing:
        arr = arr.filled(missing_big)
        print(f"已用 {missing_big} 填充缺测（转换为 ndarray 预览）")
    # 预览
    flat = arr.ravel()
    nshow = min(10, flat.size)
    print(f"前 {nshow} 个值：", flat[:nshow])
    # 缺测统计
    if is_masked:
        miss = int(arr.mask.sum()) if arr.mask is not np.ma.nomask else 0
        print("缺测个数：", miss)


def read_one_subset(bufr_file, msg_idx, subset_idx, mnemonics, missing_big, fill_missing):
    import numpy as np
    import ncepbufr as bufrmod

    b = bufrmod.open(bufr_file)

    # 尝试告知库缺测填充值（不同版本行为不同，失败也不影响后续）
    try:
        b.set_missing_value(missing_big)
    except Exception:
        try:
            b.missing_value = missing_big
        except Exception:
            pass

    # 前进到目标 message（1-based）
    cur = 0
    found_msg = False
    while b.advance() == 0:
        cur += 1
        if cur == msg_idx:
            found_msg = True
            break

    if not found_msg:
        b.close()
        raise IndexError(f"文件中未找到第 {msg_idx} 条 message。")

    # 进入目标 subset（1-based）
    for _ in range(subset_idx):
        rc = b.load_subset()
        if rc != 0:
            b.close()
            raise IndexError(f"第 {msg_idx} 条 message 不存在第 {subset_idx} 个 subset。")

    # 逐个读取（避免 tuple 行为差异）
    results = {}
    for m in mnemonics:
        try:
            arr = b.read_subset(m)  # 单独读取
            results[m] = arr
        except Exception as e:
            # 某些实现找不到变量时会抛错；记录为 None
            results[m] = None

    b.close()

    # 打印信息
    for m in mnemonics:
        print_info(m, results.get(m), fill_missing, missing_big)

    # 如果两者都存在，打印配对预览，保证没“混在一起”
    if all(results.get(m) is not None for m in mnemonics):
        a = results[mnemonics[0]]
        b = results[mnemonics[1]]
        if isinstance(a, np.ma.MaskedArray) and fill_missing:
            a = a.filled(missing_big)
        if isinstance(b, np.ma.MaskedArray) and fill_missing:
            b = b.filled(missing_big)
        va = np.ravel(np.squeeze(a))
        vb = np.ravel(np.squeeze(b))
        n = min(va.size, vb.size, 10)
        if n > 0:
            pairs = list(zip(va[:n], vb[:n]))
            print(f"示例配对 ({mnemonics[0]}, {mnemonics[1]}) 前 {n} 对：", pairs)


def main():
    require_modules()
    p = argparse.ArgumentParser(description="Robust WDIR/WSPD reader for one message/subset (ncepbufr).")
    p.add_argument("-f", "--file", default=DEFAULT_BUFR_FILE)
    p.add_argument("-m", "--msg", type=int, default=DEFAULT_MSG)
    p.add_argument("-s", "--subset", type=int, default=DEFAULT_SUBSET)
    p.add_argument("--missing-big", type=float, default=DEFAULT_MISSING_BIG)
    p.add_argument("--fill-missing", action="store_true", help="将缺测 mask 填为 --missing-big")
    args = p.parse_args()

    try:
        read_one_subset(args.file, args.msg, args.subset, MNEMONICS, args.missing_big, args.fill_missing)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
