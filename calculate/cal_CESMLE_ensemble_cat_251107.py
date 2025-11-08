'''
2025-11-7
This script is to compose the seperate files in each ensemble member folder into one single file for each ensemble member.
'''
import xarray as xr, glob, os, re, subprocess
import os, tempfile, subprocess

DATA_ROOT = "/mnt/d/"
OUTDIR = "/mnt/d/cated_CESMLE_SST/"
os.makedirs(OUTDIR, exist_ok=True)

# 从当前目录的清单提取成员标签（LE2-####.mmm）
tags = sorted({ re.sub(r'.*\.(LE2-\d{4}\.\d{3})\.pop\.h\.SST\..*', r'\1', fn)
                for fn in glob.glob("*.pop.h.SST.*.nc") })


for tag in tags:
    files = sorted(glob.glob(os.path.join(DATA_ROOT, f"*.{tag}.pop.h.SST.*.nc")))
    if not files:
        print(f"no files for {tag}")
        continue
    ds = xr.open_mfdataset(
        files, combine="by_coords", data_vars="minimal", coords="minimal",
        compat="override", chunks={"time": 120}, use_cftime=True
    )
    out = os.path.join(OUTDIR, f"SST_{tag}_1850-2100.nc")
    ds.to_netcdf(out)
    # 可选：用 CDO 再 sorttime 一下
    dirpath = os.path.dirname(out)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_sort_", suffix=".nc", dir=dirpath)
    os.close(fd)  # 先把文件描述符关掉，让 cdo 能写

    try:
        # 先写到临时文件
        subprocess.run(["cdo", "-O", "sorttimestamp", out, tmp], check=True)
        # 原子替换：把排好序的临时文件覆盖到目标文件
        os.replace(tmp, out)
    except Exception:
        # 失败时清理临时文件再抛出
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    print(f"Successfully wrote {tag}")