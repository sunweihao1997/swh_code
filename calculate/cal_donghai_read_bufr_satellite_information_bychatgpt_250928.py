#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
列出 BUFR 文件中所有唯一的卫星信息组合：
- 唯一 (SAID, SIID, SSIN) 及出现次数
- 唯一 SAID 列表
- 唯一 SIID 列表

用法:
  python list_unique_sat_info.py your_file.bufr
"""
import sys
import argparse
from collections import Counter
import numpy as np
import ncepbufr

def read_scalar_mnemonic(bufr_obj, mnemonic):
    """
    安全读取单个 mnemonic 的标量值（取本 subset 的第一项）。
    若该 mnemonic 不存在或无值，返回 None。
    """
    try:
        a = bufr_obj.read_subset(mnemonic)  # 注意: 参数必须是字符串
    except Exception:
        return None

    # 兼容旧版 dict 返回
    if isinstance(a, dict):
        v = a.get(mnemonic)
        if v is None or np.size(v) == 0:
            return None
        v0 = v[0]
        if v0 is None or (isinstance(v0, float) and np.isnan(v0)):
            return None
        try:
            return int(v0)
        except Exception:
            return float(v0)

    # 常见为 ndarray
    arr = np.asarray(a)
    if arr.size == 0:
        return None

    if arr.ndim == 1:
        v0 = arr[0]
    else:
        # 2D 时通常形如 (nrows, ncols)
        v0 = arr[0, 0]

    if v0 is None or (isinstance(v0, float) and np.isnan(v0)):
        return None

    try:
        return int(v0)
    except Exception:
        return float(v0)

def main():
    ap = argparse.ArgumentParser(description="List unique (SAID, SIID, SSIN) from a BUFR file.")
    ap.add_argument("bufr_path", help="Path to BUFR file")
    args = ap.parse_args()

    b = ncepbufr.open(args.bufr_path)

    # 统计容器
    triplet_counter = Counter()   # (SAID, SIID, SSIN) -> count
    unique_said = set()
    unique_siid = set()
    unique_ssin = set()

    msg_idx = 0
    # 遍历 message
    while b.advance() == 0:
        msg_idx += 1
        # 遍历 subset
        while b.load_subset() == 0:
            said = read_scalar_mnemonic(b, "SAID")
            siid = read_scalar_mnemonic(b, "SIID")
            ssin = read_scalar_mnemonic(b, "SSIN")

            # 更新统计
            triplet_counter[(said, siid, ssin)] += 1
            if said is not None: unique_said.add(int(said))
            if siid is not None: unique_siid.add(int(siid))
            if ssin is not None: unique_ssin.add(int(ssin))

    b.close()

    # 输出结果
    print("\n=== 唯一 (SAID, SIID, SSIN) 组合（按 SAID,SIID,SSIN 排序） ===")
    def sort_key(item):
        (said, siid, ssin), cnt = item
        # None 排到最后
        return (
            (1e12 if said is None else said),
            (1e12 if siid is None else siid),
            (1e12 if ssin is None else ssin)
        )
    for (said, siid, ssin), cnt in sorted(triplet_counter.items(), key=sort_key):
        print(f"(SAID={said}, SIID={siid}, SSIN={ssin})  count={cnt}")

    print("\n=== 唯一 SAID（卫星） ===")
    if unique_said:
        print(", ".join(str(x) for x in sorted(unique_said)))
    else:
        print("(无)")

    print("\n=== 唯一 SIID（仪器） ===")
    if unique_siid:
        print(", ".join(str(x) for x in sorted(unique_siid)))
    else:
        print("(无)")

    print("\n=== 唯一 SSIN（传感器指示，如文件未提供可能为空） ===")
    if unique_ssin:
        print(", ".join(str(x) for x in sorted(unique_ssin)))
    else:
        print("(无)")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("用法: python list_unique_sat_info.py <your_satellite_file.bufr>")
        sys.exit(1)
    main()
