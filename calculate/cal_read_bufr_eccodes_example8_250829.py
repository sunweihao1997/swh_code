# -*- coding: utf-8 -*-
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer, FlatJsonRenderer

bufr_path = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"

def split_bufr_messages(blob: bytes):
    """
    在原始二进制中按 BUFR 消息切分，返回每条消息的 bytes。
    依据：'BUFR' + 3 字节总长度 + 1 字节版本号（section 0）
    总长度为整个 BUFR 消息的长度（从 'BUFR' 到末尾 '7777'）
    """
    msgs = []
    i = 0
    while True:
        j = blob.find(b"BUFR", i)
        if j < 0:
            break
        if j + 8 > len(blob):
            break  # 不足以读总长与版本号
        total_len = int.from_bytes(blob[j+4:j+7], "big")
        k = j + total_len
        if k > len(blob):  # 文件尾部不完整则终止
            break
        msgs.append(blob[j:k])
        i = k
    return msgs

# --- 主流程 ---
dec = Decoder()
r_flat = FlatTextRenderer()
r_json = FlatJsonRenderer()

with open(bufr_path, "rb") as f:
    blob = f.read()

messages = split_bufr_messages(blob)
if not messages:
    # 如果没切出多条，就当作只有一条
    messages = [blob]

print(f"检测到 {len(messages)} 条 BUFR 消息\n")

for idx, mbytes in enumerate(messages, 1):
    print("="*14, f" Message {idx} ", "="*14)
    try:
        # 某些版本的 process 返回单个 BufrMessage（不是可迭代）
        msg = dec.process(mbytes)

        # 扁平文本视图（等价 CLI: pybufrkit decode）
        flat_view = r_flat.render(msg)
        print("\n-- Flat view --")
        print(flat_view)

        # 简单判断：是否像 DX（本地表定义）消息
        if ("Table A:" in flat_view) or ("TABLE A:" in flat_view):
            print(">> 这条很像“DX（本地表定义）消息”，通常不含真实观测值。")

        # 扁平 JSON（等价 CLI: pybufrkit decode -j）
        print("\n-- Flat JSON --")
        print(r_json.render(msg))

    except Exception as e:
        print(f"[ERROR] 这条消息解析失败：{e}")

print("\n说明：如果看到 “Cannot find sub-centre ... Local table not in use.” 的告警，"
      "意思是 pybufrkit 没有找到该子中心的本地表，只用 WMO 公共表来解码。"
      "而 NESDIS SATWND 往往依赖本地表，所以很多字段会只显示‘描述’，没有真实数值。")
