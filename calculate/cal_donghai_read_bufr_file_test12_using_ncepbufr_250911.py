from ncepbufr import open as bufr_open
import numpy as np

BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
TARGET_FIELDS = ["WDIR", "WSPD", "CLATH", "CLONH"]

bufr = bufr_open(BUFR_FILE)

imsg = 0
found_messages = []

while bufr.advance() == 0:
    imsg += 1
    nsub = bufr.subsets

    if nsub == 0:
        continue

    hits = {f: 0 for f in TARGET_FIELDS}

    # 读取每个 subset，统计字段命中数
    isub = 0
    while bufr.load_subset() == 0:
        isub += 1
        data = bufr.read_subset(" ".join(TARGET_FIELDS))
        if data is None:
            continue
        values = np.array(data).flatten()
        for i, f in enumerate(TARGET_FIELDS):
            if i < len(values) and values[i] is not None and values[i] != 1e11:
                hits[f] += 1

    # 统计是否符合要求（都至少命中1个）
    if all(hits[f] > 0 for f in TARGET_FIELDS):
        print(f"\n✅ 找到符合条件的 message #{imsg}, subsets={nsub}")
        print("字段命中数：", hits)
        found_messages.append(imsg)

bufr.close()

if not found_messages:
    print("\n⚠️ 没找到同时含 WDIR/WSPD/CLATH/CLONH 的 message")
else:
    print("\n🎯 共找到这些 message:", found_messages)
