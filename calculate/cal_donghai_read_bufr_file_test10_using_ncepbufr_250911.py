import ncepbufr
import numpy as np

BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"

# 想读取的第几个 message（从 1 开始计数）
target_msg = 15810

# 想读取的变量（可改）
fields = ["WDIR", "WSPD", "UWND", "VWND", "CLATH", "CLONH", "PRLC"]
field_expr = " ".join(fields)

bufr = ncepbufr.open(BUFR_FILE)

msg_index = 0
found = False
rows = []  # 用来收集该 message 下所有 subset 的记录

while bufr.advance() == 0:
    msg_index += 1

    if msg_index < target_msg:
        continue
    if msg_index > target_msg:
        break  # 已经过了目标 message，直接结束

    # 到了目标 message，开始逐个 subset 读取
    msg_type = bufr.msg_type
    nsub = bufr.subsets
    print(f"👉 到达目标 Message #{msg_index}: type={msg_type}, subsets={nsub}")

    isub = 0
    while bufr.load_subset() == 0:
        isub += 1
        try:
            data = bufr.read_subset(field_expr)
        except Exception as e:
            # 该 subset 不含这些要素或解码失败，跳过
            continue

        if data is None or getattr(data, "size", 0) == 0:
            continue

        # read_subset 通常返回结构化数组；统一成 (N, ) 的迭代形式
        data = np.atleast_1d(data)

        # 将结构化数组的每条记录转成普通 dict，便于后续处理/打印
        for rec in data:
            row = {}
            for name in fields:
                try:
                    val = rec[name]
                    # 取出标量值
                    if hasattr(val, "item"):
                        val = val.item()
                except Exception:
                    val = None
                row[name] = val
            row["_subset_idx"] = isub
            rows.append(row)

    found = True
    break  # 只读这一个 message

bufr.close()

if not found:
    print(f"❌ 文件中没有第 {target_msg} 个 message（总数 < {target_msg}）。")
else:
    if not rows:
        print(f"⚠️ 第 {target_msg} 个 message 中未在任何 subset 里读取到 {fields} 这些要素。")
    else:
        # 简要预览前几行
        print(f"\n✅ 第 {target_msg} 个 message 读取到 {len(rows)} 条记录（来自多个 subset）：")
        preview = rows[:10]
        # 美观打印
        from pprint import pprint
        pprint(preview)
        # 如需保存成 CSV，取消下面两行注释
        # import pandas as pd
        # pd.DataFrame(rows).to_csv(f"message_{target_msg}_extract.csv", index=False)
        # print(f"\n已保存到 message_{target_msg}_extract.csv")
