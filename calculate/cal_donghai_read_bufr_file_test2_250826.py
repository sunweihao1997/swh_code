'''
2025-8-26
This script is used to read the BUFR file and convert it to CSV file.

'''
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer, NestedTextRenderer

decoder = Decoder()
with open('/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr','rb') as f:
    msg = decoder.process(f.read())

print(FlatTextRenderer().render(msg))