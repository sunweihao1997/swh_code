import ncepbufr
import numpy as np

BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
MISSING = 100000000000.0
MNEMS = ["WDIR", "WSPD", "UWND", "VWND", "CLATH", "CLONH", "PRLC"]

bufr = ncepbufr.open(BUFR_FILE)

msg_index = 0
while bufr.advance() == 0:
    msg_index += 1
    nsub = bufr.subsets

    # 找到第一条有 subset 的 message
    if nsub > 0:
        print(f"Message #{msg_index}: type={bufr.msg_type}, subsets={nsub}")

        isub = 0
        while bufr.load_subset() == 0:
            isub += 1
            data = bufr.read_subset(" ".join(MNEMS))
            if data is None or data.size == 0:
                continue

            vals = np.asarray(data).reshape(-1)
            # 全部是缺测就跳过
            if all((isinstance(v, float) and v == MISSING) for v in vals):
                continue

            print(f"\nMessage #{msg_index}, Subset #{isub}")
            for name, v in zip(MNEMS, vals):
                if isinstance(v, float) and v == MISSING:
                    print(f"  {name}: --")
                else:
                    print(f"  {name}: {v}")

        break  # 只看这一条 message
bufr.close()
