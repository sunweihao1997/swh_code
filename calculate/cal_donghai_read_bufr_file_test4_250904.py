'''
2025-9-4
This script is try to read BUFR file through pybufrkit.

'''
import re
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer

bufr_path = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"

# 读入二进制
with open(bufr_path, "rb") as f:
    data = f.read()

dec = Decoder()
res = dec.process(data)

# 兼容：单条 or 多条消息
try:
    messages = list(res)
except TypeError:
    messages = [res]

print(f"Total messages: {len(messages)}")

r = FlatTextRenderer()

def render_to_text(msg):
    """把一条 BUFR message 渲染成扁平文本"""
    return r.render(msg)

def take_subset1_block(flat_text):
    """
    从扁平文本里切出“subset 1 of N”的那一段。
    如果找不到，就返回整个文本（尽量不让流程中断）。
    """
    # 典型标记：###### subset 1 of 255 ######
    m = re.search(r"######\s*subset\s*1\s*of\s*\d+\s*######", flat_text)
    if not m:
        return flat_text
    start = m.end()
    # 到下一个 "###### subset 2 ..." 或到 section 5 结束
    m2 = re.search(r"######\s*subset\s*2\s*of\s*\d+\s*######|<<<<<< section 5 >>>>>>", flat_text)
    end = m2.start() if m2 else len(flat_text)
    return flat_text[start:end]

def pick_first_number(s):
    """
    尝试从一行中提取一个“像数值”的东西（整数/小数/科学计数）。
    如果没有就返回 None。
    """
    # 优先匹配“= 数字”或“: 数字”
    m = re.search(r"[=:]\s*([\-+]?\d+(\.\d+)?([eE][\-+]?\d+)?)", s)
    if m:
        return m.group(1)
    # 不行就抓行尾可能的数字
    m2 = re.search(r"([\-+]?\d+(\.\d+)?([eE][\-+]?\d+)?)\s*$", s)
    return m2.group(1) if m2 else None

# 关注的“元素短名”及别名列表（同一物理量的不同编码）
wanted = {
    # 时间
    "YEAR": ["YEAR"], "MNTH": ["MNTH"], "DAYS": ["DAYS"],
    "HOUR": ["HOUR"], "MINU": ["MINU"], "SECO": ["SECO"],

    # 经纬度（粗/高精度）
    "CLAT": ["CLAT", "CLATH"],  # 纬度
    "CLON": ["CLON", "CLONH"],  # 经度

    # 垂直/高度/气压
    "PRLC": ["PRLC"],           # 压力 (Pa)
    "HGHT": ["HGHT","HMSL","HOLS","HITE"],  # 各种高度/海拔/位势高

    # 风
    "WDIR": ["WDIR"],
    "WSPD": ["WSPD"],
    "UWND": ["UWND","UMWV"],
    "VWND": ["VWND","VWMV"],
}

for i, msg in enumerate(messages, 1):
    print(f"\n=== Message {i} ===")
    flat = render_to_text(msg)

    # 打印一小段头信息（可选）
    print("---- Section 1 quick peek ----")
    s1 = msg.sections[1]
    # 某些版本把字段封装为对象，做兼容处理
    def val(section, key):
        try:
            v = section._namespace.get(key)
            return getattr(v, "value", v)
        except Exception:
            return None
    print("edition =", getattr(msg, "edition", None))
    print("master_table_version =", val(s1, "master_table_version"))
    print("local_table_version =",  val(s1, "local_table_version"))

    # 拿 subset 1 的文本块
    block = take_subset1_block(flat)

    # 为了易读，先打印前 40 行看看（你也可以注释掉）
    head_preview = "\n".join(block.splitlines()[:40])
    print("\n---- Subset 1 block (head) ----")
    print(head_preview)

    # 逐行查找我们关心的短名（如 "WDIR", "WSPD", "CLAT"...）
    found = {}
    for line in block.splitlines():
        # 典型行里会有短名，例如：
        # " 783 000013 ELEMENT NAME, LINE 1 b'WDIR ...'"
        # 或者标准行：
        # " 785 000013 ELEMENT NAME, LINE 1  WDIR WIND DIRECTION ...  270"
        # 甚至本地表定义时会出现 "ELEMENT NAME" 行，我们也匹配短名字符串
        for std_key, aliases in wanted.items():
            for alias in aliases:
                # 只要这一行包含这个短名（大小写敏感，BUFR里通常大写）
                if re.search(rf"\b{alias}\b", line):
                    num = pick_first_number(line)
                    if num is not None and std_key not in found:
                        found[std_key] = num
                        break  # 这个 std_key 已经拿到一个值了，换下一个 std_key

    # 打印结果
    print("\n---- Extracted values from subset 1 ----")
    if not found:
        print("（没从 subset 1 的文本里识别出任何目标量。可能：这条消息没有观测数值，或当前本地表/模板导致只显示结构。）")
    else:
        for k in ["YEAR","MNTH","DAYS","HOUR","MINU","SECO",
                  "CLAT","CLON","PRLC","HGHT","WDIR","WSPD","UWND","VWND"]:
            if k in found:
                print(f"{k}: {found[k]}")
