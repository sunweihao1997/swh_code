'''
Try use different table to read BUFR file
'''

import eccodes as ec

fname = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"

trial_versions = [44, 36, 33, 27, 24, 19, 14, 13]  # 从新到旧试
ok = 0
fail = 0

with open(fname, "rb") as f:
    msg_no = 0
    while True:
        gid = ec.codes_bufr_new_from_file(f)
        if gid is None:
            break
        msg_no += 1
        nsub = ec.codes_get(gid, "numberOfSubsets")
        print(f"\n=== Message {msg_no} ===")
        print("numberOfSubsets =", nsub)

        if nsub == 0:
            print("→ 子集为 0，跳过")
            ec.codes_release(gid)
            continue

        # 逐个版本尝试解包
        unpacked = False
        last_err = None
        for v in trial_versions:
            try:
                # 先复位（防止上次尝试残留）
                try:
                    ec.codes_set(gid, "unpack", 0)
                except Exception:
                    pass
                ec.codes_set(gid, "masterTablesVersionNumber", v)
                # 有些报文没有本地表：把 local 置 0 更稳妥
                ec.codes_set(gid, "localTablesVersionNumber", 0)
                ec.codes_set(gid, "unpack", 1)
                print(f"✓ 解包成功，使用 masterTablesVersionNumber={v}")
                unpacked = True
                break
            except Exception as e:
                last_err = e

        if not unpacked:
            print(f"⚠️  解包失败（可能缺表/用本地表）：{last_err}")
            fail += 1
            ec.codes_release(gid)
            continue

        ok += 1
        # —— 在这里读真正的观测值（能读到多少取决于模板）——
        def safe_get_array(key):
            try:
                return ec.codes_get_array(gid, key)
            except Exception:
                return None

        lat = safe_get_array("latitude")
        lon = safe_get_array("longitude")
        ws  = safe_get_array("windSpeed")
        wd  = safe_get_array("windDirection")

        if lat is not None and lon is not None:
            n = min(len(lat), len(lon), len(ws or []), len(wd or [])) if ws is not None and wd is not None else len(lat)
            print(f"观测点数(可输出) ≈ {n}")
            for i in range(min(n, 5)):  # 只示例前5条
                s_ws = f"{ws[i]:.1f} m/s" if ws is not None else "NA"
                s_wd = f"{wd[i]:.1f} deg" if wd is not None else "NA"
                print(f"  Obs {i+1}: lat={lat[i]:.3f}, lon={lon[i]:.3f}, ws={s_ws}, wd={s_wd}")
        else:
            print("已解包，但此模板不是风场（可能是辐射/亮温），需要换一套 keys 读取。")

        ec.codes_release(gid)

print(f"\n完成：成功 {ok} 条，失败 {fail} 条。")
