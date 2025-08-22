#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查目录下的 ERA5 NetCDF 文件是否能正常打开
- 递归扫描目录里的 .nc 文件
- 尝试用 xarray 打开
- 无法打开的文件收集起来，最后统一输出
"""

import sys
from pathlib import Path
import xarray as xr

def files_under(root: Path):
    for p in root.rglob("*.nc"):
        if p.is_file():
            yield p

def can_open_nc(p: Path) -> bool:
    if p.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(p) as ds:
            _ = list(ds.variables)[:1]  # 触发一次读取，避免惰性假成功
        return True
    except Exception:
        return False

def main():
    if len(sys.argv) != 2:
        print("用法: python check_nc.py <目录路径>")
        sys.exit(2)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"[错误] 路径不存在: {root}")
        sys.exit(2)

    bad = []
    total = 0
    ok = 0

    for f in sorted(files_under(root)):
        total += 1
        if can_open_nc(f):
            ok += 1
            print(f"[OK]  {f}")
        else:
            bad.append(str(f))
            print(f"[BAD] {f}")

    print("\n==== 总结 ====")
    print(f"总文件数: {total}")
    print(f"可打开数: {ok}")
    print(f"不可打开数: {len(bad)}")

    if bad:
        print("\n==== 无法打开的文件 ====")
        for p in bad:
            print(p)

    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()