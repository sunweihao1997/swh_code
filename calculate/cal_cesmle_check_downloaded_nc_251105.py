#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
20251105
This script checks the integrity of NetCDF files in a specified directory or file.
'''
import argparse
import csv
import datetime as dt
import os
import sys
import subprocess
from typing import List, Tuple, Optional

# 依赖：netCDF4
try:
    from netCDF4 import Dataset
except Exception as e:
    print("缺少依赖：netCDF4。请先安装： pip install netCDF4", file=sys.stderr)
    raise


def find_nc_files(root: str, recursive: bool) -> List[str]:
    files: List[str] = []
    if os.path.isfile(root):
        if root.lower().endswith((".nc", ".cdf", ".nc4", ".netcdf")):
            files.append(os.path.abspath(root))
        return files

    if not recursive:
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isfile(p) and p.lower().endswith((".nc", ".cdf", ".nc4", ".netcdf")):
                files.append(os.path.abspath(p))
    else:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith((".nc", ".cdf", ".nc4", ".netcdf")):
                    files.append(os.path.abspath(os.path.join(dirpath, name)))
    return sorted(files)

def try_ncdump_header(path: str, timeout: int = 30) -> Tuple[bool, Optional[str]]:
    """使用 ncdump -h 检查（如果系统有 ncdump）。"""
    try:
        # 检查 ncdump 是否存在
        res = subprocess.run(["ncdump", "-h", path],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             timeout=timeout)
        if res.returncode == 0:
            return True, None
        else:
            msg = (res.stderr or res.stdout).decode(errors="replace").strip()
            return False, f"ncdump 失败：{msg}"
    except FileNotFoundError:
        return True, None  # 没有 ncdump，不算失败
    except subprocess.TimeoutExpired:
        return False, "ncdump 超时"
    except Exception as e:
        return False, f"ncdump 异常：{e}"

def read_small_slice(var) -> None:
    """读取变量的一个小切片/标量，用于触发实际 IO。"""
    try:
        shape = getattr(var, "shape", None)
        if shape is None:
            # 没有 shape（极少见），尝试直接读取
            _ = var[:]
            return
        if len(shape) == 0:
            # 标量变量
            _ = var[()]
            return
        if any(dim == 0 for dim in shape):
            # 含零长度维度，读取将是空数组，但仍尝试一次读取
            _ = var[:]
            return
        # 针对普通数组，构造一个小切片（每个维度取前 1 个元素）
        index = tuple(slice(0, 1) for _ in shape)
        _ = var[index]
    except Exception:
        # 抛出让上层捕获并记录
        raise

def deep_read_variable(var, chunk_elements: int = 2_000_000) -> None:
    """
    深度检查：尽量把整个变量读一遍。为避免内存暴涨，按元素数大致分块。
    chunk_elements 是每次预计读取的最大元素数，实际按第一个维度切块。
    """
    shape = getattr(var, "shape", None)
    if shape is None or len(shape) == 0:
        # 标量
        _ = var[()]
        return
    if any(d == 0 for d in shape):
        _ = var[:]
        return

    # 估算单位元素大小（字节），用于决定块大小
    try:
        import numpy as np
        dtype_size = np.dtype(var.dtype).itemsize
    except Exception:
        dtype_size = 8  # 估个保守值

    # 以第 0 维为主分块
    n0 = shape[0]
    # 其余维度的元素数
    rest = 1
    for d in shape[1:]:
        rest *= d
    # 计算每块在第0维可读取的步长
    max_rows = max(1, chunk_elements // max(1, rest))
    start = 0
    while start < n0:
        stop = min(n0, start + max_rows)
        slicer = (slice(start, stop),) + tuple(slice(None) for _ in shape[1:])
        _ = var[slicer]
        start = stop

def check_one_file(path: str, deep: bool, use_ncdump: bool) -> Tuple[str, str]:
    """
    返回 (status, message)
    status: OK / FAIL
    message: 失败时的原因；成功时包含基本信息
    """
    # 先用 ncdump（如可用）
    if use_ncdump:
        ok, msg = try_ncdump_header(path)
        if not ok:
            return "FAIL", f"{msg}"

    # 尝试用 netCDF4 打开
    try:
        ds = Dataset(path, mode="r")
    except Exception as e:
        return "FAIL", f"无法打开（netCDF4）：{e}"

    # 读取全局属性
    try:
        _ = {k: getattr(ds, k) for k in ds.ncattrs()}
    except Exception as e:
        try:
            ds.close()
        except Exception:
            pass
        return "FAIL", f"读取全局属性失败：{e}"

    # 遍历变量并做读取测试
    try:
        for vname, var in ds.variables.items():
            # 先小切片触发读
            read_small_slice(var)
            # 深度检查则全量按块读取
            if deep:
                deep_read_variable(var)
    except Exception as e:
        try:
            ds.close()
        except Exception:
            pass
        return "FAIL", f"变量 `{vname}` 读取失败：{e}"

    # 关闭文件
    try:
        ds.close()
    except Exception as e:
        return "FAIL", f"关闭文件失败：{e}"

    return "OK", "打开与读取测试通过"

def human_size(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(nbytes)
    for u in units:
        if x < 1024.0:
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{x:.1f} PB"

def main():
    parser = argparse.ArgumentParser(
        description="批量检查 NetCDF 文件完整性并输出 CSV 报告"
    )
    parser.add_argument("path", help="文件或目录路径")
    parser.add_argument("--recursive", action="store_true", help="递归扫描目录")
    parser.add_argument("--deep", action="store_true", help="深度检查（完整读取变量，较慢）")
    parser.add_argument("--no-ncdump", action="store_true",
                        help="不使用 ncdump -h 进行额外校验")
    parser.add_argument("--output", default="netcdf_integrity_report.csv",
                        help="输出 CSV 文件路径（默认：netcdf_integrity_report.csv）")
    args = parser.parse_args()

    files = find_nc_files(args.path, args.recursive)
    if not files:
        print("未找到 NetCDF 文件（后缀 .nc/.cdf/.nc4/.netcdf）。", file=sys.stderr)
        sys.exit(2)

    print(f"发现 {len(files)} 个文件，开始检查……")
    ok_count = 0
    fail_count = 0
    rows = []

    for i, f in enumerate(files, 1):
        try:
            st = os.stat(f)
            size = st.st_size
            mtime = dt.datetime.fromtimestamp(st.st_mtime)
        except Exception:
            size = None
            mtime = None

        status, msg = check_one_file(f, deep=args.deep, use_ncdump=not args.no_ncdump)
        if status == "OK":
            ok_count += 1
        else:
            fail_count += 1

        rows.append({
            "file": f,
            "status": status,
            "message": msg,
            "size_bytes": size if size is not None else "",
            "size_human": human_size(size) if size is not None else "",
            "mtime": mtime.isoformat(sep=" ", timespec="seconds") if mtime else "",
        })

        print(f"[{i}/{len(files)}] {status} - {os.path.basename(f)}"
              + (f" - {msg}" if status != "OK" else ""))

    # 写 CSV 报告
    fieldnames = ["file", "status", "message", "size_bytes", "size_human", "mtime"]
    try:
        with open(args.output, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    except Exception as e:
        print(f"写入报告失败：{e}", file=sys.stderr)
        sys.exit(3)

    print(f"\n完成。OK: {ok_count}，FAIL: {fail_count}。")
    print(f"报告已保存：{args.output}")

    # 若存在失败，返回非零退出码，方便在 CI/脚本中使用
    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
