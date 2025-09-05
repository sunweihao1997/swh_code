'''
Skip data which can not be found in eccodes
'''

import eccodes

filename = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"

with open(filename, "rb") as f:
    msg_count = 0
    while True:
        try:
            gid = eccodes.codes_bufr_new_from_file(f)
        except Exception:
            break  # 文件读完了

        if gid is None:
            break

        msg_count += 1
        print(f"\n=== Message {msg_count} ===")

        # 获取子集数
        nsub = eccodes.codes_get(gid, "numberOfSubsets")
        print("numberOfSubsets =", nsub)

        # 如果没有观测，就跳过
        if nsub == 0:
            eccodes.codes_release(gid)
            continue

        try:
            # 解包数据
            eccodes.codes_set(gid, "unpack", 1)

            # 拿几个常见变量试试
            lat = eccodes.codes_get_array(gid, "latitude")
            lon = eccodes.codes_get_array(gid, "longitude")
            ws  = eccodes.codes_get_array(gid, "windSpeed")
            wd  = eccodes.codes_get_array(gid, "windDirection")

            for i in range(len(lat)):
                print(f"Obs {i+1}: lat={lat[i]:.2f}, lon={lon[i]:.2f}, "
                      f"ws={ws[i]:.1f} m/s, wd={wd[i]:.1f} deg")

        except Exception as e:
            print("⚠️  这一条 message 解码失败，可能用了本地表:", e)

        eccodes.codes_release(gid)
