import ncepbufr
import numpy as np
import xarray as xr

# 参数
bufr_file = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
msg_idx   = 1000   # 你想要测试的第几条 message
out_file  = "/home/sun/data/bufr_test/message_1000.nc"

# 打开 BUFR 文件
b = ncepbufr.open(bufr_file)

# 定位到第 msg_idx 条 message
cur = 0
while b.advance() == 0:
    cur += 1
    if cur == msg_idx:
        break

# 遍历 subsets
obs_list = []
while b.load_subset() == 0:
    try:
        rec = b.read_subset("CLAT CLON WDIR WSPD")
        # rec 是 MaskedArray，shape = (nvar, nvals) 或 (nvals,)；依赖 subset 定义
        obs = {
            "CLAT": np.ravel(rec[0]),
            "CLON": np.ravel(rec[1]),
            "WDIR": np.ravel(rec[2]),
            "WSPD": np.ravel(rec[3]),
        }
        obs_list.append(obs)
    except Exception as e:
        print("subset 读取失败:", e)

b.close()

# 拼接成统一数组
n = len(obs_list)
clat = np.array([o["CLAT"][0] if len(o["CLAT"]) else np.nan for o in obs_list])
clon = np.array([o["CLON"][0] if len(o["CLON"]) else np.nan for o in obs_list])
wdir = np.array([o["WDIR"][0] if len(o["WDIR"]) else np.nan for o in obs_list])
wspd = np.array([o["WSPD"][0] if len(o["WSPD"]) else np.nan for o in obs_list])

# 转成 xarray.Dataset
ds = xr.Dataset(
    {
        "WDIR": ("obs", wdir),
        "WSPD": ("obs", wspd),
    },
    coords={
        "obs": np.arange(n),
        "CLAT": ("obs", clat),
        "CLON": ("obs", clon),
    },
    attrs={
        "source": f"BUFR message {msg_idx}",
    }
)

# 写出 NetCDF
ds.to_netcdf(out_file)
print(f"[OK] 已写出 {out_file}, 包含 {n} 条观测")
