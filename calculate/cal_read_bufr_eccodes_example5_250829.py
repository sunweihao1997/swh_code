import eccodes as ec

path = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"

with open(path, "rb") as f:
    msg_index = 0
    while True:
        ibufr = ec.codes_bufr_new_from_file(f)
        if ibufr is None:   # 文件读完
            break

        msg_index += 1
        # 一定要先 unpack 再迭代 keys
        ec.codes_set(ibufr, "unpack", 1)

        print(f"\n=== Message {msg_index} ===")
        # 打个基本信息，确认数据确实被解开
        try:
            nsub = ec.codes_get(ibufr, "numberOfSubsets")
            print("numberOfSubsets:", nsub)
        except Exception as e:
            print("cannot read numberOfSubsets:", e)

        # 用 BUFR 专用的 keys 迭代器
        it = ec.codes_bufr_keys_iterator_new(ibufr)
        try:
            while ec.codes_bufr_keys_iterator_next(it):
                key = ec.codes_bufr_keys_iterator_get_name(it)

                # 跳过属性字段(带 #)；只要“主值”
                if "#" in key:
                    continue

                try:
                    size = ec.codes_get_size(ibufr, key)
                    if size > 1:
                        val = ec.codes_get_array(ibufr, key)
                    else:
                        val = ec.codes_get(ibufr, key)
                    print(key, "=>", val)
                except ec.KeyValueNotFoundError:
                    # 有些 key 对某些模板/子集不存在，直接略过
                    pass
        finally:
            ec.codes_bufr_keys_iterator_delete(it)
            ec.codes_release(ibufr)
