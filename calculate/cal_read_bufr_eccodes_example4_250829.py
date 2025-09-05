import eccodes

f = open("/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr", "rb")

ibufr = eccodes.codes_bufr_new_from_file(f)

#value = eccodes.codes_get(ibufr, "dataCategory")
eccodes.codes_set(ibufr, "unpack", 1)

# 遍历 keys
it = eccodes.codes_keys_iterator_new(ibufr, "data")  # "data" 命名空间
while eccodes.codes_keys_iterator_next(it):
    key = eccodes.codes_keys_iterator_get_name(it)
    print(key)
eccodes.codes_keys_iterator_delete(it)

eccodes.codes_release(ibufr)

f.close()