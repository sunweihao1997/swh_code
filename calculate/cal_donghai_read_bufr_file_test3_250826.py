'''
2025-8-26
This script is used to read the BUFR file and convert it to CSV file.

'''
import json

SOME_BUFR_FILE = '/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr'
# Decode a BUFR file
from pybufrkit.decoder import Decoder
decoder = Decoder()
with open(SOME_BUFR_FILE, 'rb') as ins:
    bufr_message = decoder.process(ins.read())


## Convert the BUFR message to JSON
#from pybufrkit.renderer import FlatJsonRenderer
#json_data = FlatJsonRenderer().render(bufr_message)
#
#
#def default(o):
#    if isinstance(o, (bytes, bytearray)):
#        try:
#            return o.decode("utf-8", errors="replace")  # 常见BUFR字符串字段
#        except Exception:
#            return o.hex()  # 兜底：不可解码的原始字节用hex表示
#    # 需要时可在这里继续支持别的非常见类型（如 numpy 类型）
#    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
#
#with open("/mnt/f/output.json", "w", encoding="utf-8") as f:
#    json.dump(json_data, f, ensure_ascii=False, indent=2, default=default)

# Encode the JSON back to BUFR file
#from pybufrkit.encoder import Encoder
#encoder = Encoder()
#exit()
#
#bufr_message_new = encoder.process(json_data)
#with open(BUFR_OUTPUT_FILE, 'wb') as outs:
#    outs.write(bufr_message_new.serialized_bytes)
#
#
#
## Decode for multiple messages from a single file
#from pybufrkit.decoder import generate_bufr_message
#with open(SOME_FILE, 'rb') as ins:
#    for bufr_message in generate_bufr_message(decoder, ins.read()):
#        pass  # do something with the decoded message object
#
# Query the metadata
#from pybufrkit.mdquery import MetadataExprParser, MetadataQuerent
#n_subsets = MetadataQuerent(MetadataExprParser()).query(bufr_message, '%n_subsets')
#
#print(n_subsets)

# Query the data
from pybufrkit.dataquery import NodePathParser, DataQuerent
query_result = DataQuerent(NodePathParser()).query(bufr_message, '001002')
print(query_result)

for node in query_result:      # 遍历每个匹配节点
    print(node.values)         # node.values 通常是一个(扁平)列表，含该节点的值
# 或直接一次性看拍平的值列表：
print(query_result.values)     # 常用来快速取“所有值”，便于拼成 CSV 列

# Script
#from pybufrkit.script import ScriptRunner
## NOTE: must use the function version of print (Python 3), NOT the statement version
#code = """print('Multiple' if ${%n_subsets} > 1 else 'Single')"""
#runner = ScriptRunner(code)
#runner.run(bufr_message)