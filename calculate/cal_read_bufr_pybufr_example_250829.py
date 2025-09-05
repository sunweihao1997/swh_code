from pybufrkit.decoder import Decoder

decoder = Decoder()
with open("/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr", "rb") as f:
    messages = decoder.process(f.read())

for msg in messages:
    for subset in msg.subsets:
        print(subset)  # 这里才是真正的观测数据
