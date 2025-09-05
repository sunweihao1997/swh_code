from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer, FlatJsonRenderer

bufr_file = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"

dec = Decoder()
r_flat = FlatTextRenderer()           # 扁平文本（类似 `pybufrkit decode`）
r_json = FlatJsonRenderer()           # 扁平 JSON（类似 `pybufrkit decode -j`）

with open(bufr_file, "rb") as f:
    data = f.read()  # 必须读成 bytes

for i, msg in enumerate(dec.process(data), 1):
    print(f"\n=== Message {i} ===")

    # 1) 扁平文本
    flat_text = r_flat.render(msg)
    print("\n-- Flat view --")
    print(flat_text)

    # 2) 扁平 JSON
    print("\n-- Flat JSON --")
    print(r_json.render(msg))

    # 小检测：是否像 DX（表定义）消息
    if "Table A:" in flat_text or "TABLE A:" in flat_text:
        print(">> 这个 message 很可能是“表定义（DX）消息”，没有实际观测值。")
