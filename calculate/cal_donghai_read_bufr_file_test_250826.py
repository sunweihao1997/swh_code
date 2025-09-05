'''
2025-8-26
This script is used to read the BUFR file and convert it to CSV file.

'''
from pybufrkit.decoder import Decoder
from pybufrkit.dataquery import NodePathParser, DataQuerent

decoder = Decoder()
with open('/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr', 'rb') as ins:
    bufr_message = decoder.process(ins.read())

from pybufrkit.dataquery import NodePathParser, DataQuerent
query_result = DataQuerent(NodePathParser()).query(bufr_message, '001002')

print(query_result)