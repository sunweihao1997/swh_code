import eccodes

f = open("/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr", "rb")

while True:
    try:
        gid = eccodes.codes_bufr_new_from_file(f)
    except Exception:
        break
    if gid is None:
        break

    eccodes.codes_set(gid, "unpack", 1)  # 必须先 unpack 才能读值

    # 列出所有 key
    keys = eccodes.codes_keys_iterator_new(gid)
    while eccodes.codes_keys_iterator_next(keys):
        keyname = eccodes.codes_keys_iterator_get_name(keys)
        value = eccodes.codes_get(gid, keyname)
        print(keyname, "=", value)

    eccodes.codes_release(gid)

f.close()
