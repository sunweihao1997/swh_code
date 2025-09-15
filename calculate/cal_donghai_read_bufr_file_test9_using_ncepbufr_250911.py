import ncepbufr, os

BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
TARGET_INDEX = 15000
TMP_FILE = "dump_table_output.txt"

bufr = ncepbufr.open(BUFR_FILE)

imsg = 0
while bufr.advance() == 0:
    imsg += 1
    if imsg != TARGET_INDEX:
        continue

    print(f"Message #{imsg}: type = NC{bufr.msg_type}, subsets = {bufr.subsets}")

    # ⚡ 将 table dump 到临时文件
    bufr.dump_table(TMP_FILE)

    # 读回内容并打印
    with open(TMP_FILE, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    print("\n=== 描述符模板（Table B 名） ===")
    for ln in lines:
        print(ln)

    # 清理临时文件
    os.remove(TMP_FILE)

    break

bufr.close()
