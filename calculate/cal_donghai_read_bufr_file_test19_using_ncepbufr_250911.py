#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
TARGET_MSG = 15658

try:
    import ncepbufr as bufrmod
except ImportError:
    print("请先安装 ncepbufr：pip install ncepbufr", file=sys.stderr)
    sys.exit(1)

b = bufrmod.open(BUFR_FILE)

cur = 0
found = False
while b.advance() == 0:
    cur += 1
    if cur == TARGET_MSG:
        # 依次尝试 subset / msg_type 属性 / msg_type() 方法
        subset_tag = getattr(b, "subset", None)
        msg_type_attr = getattr(b, "msg_type", None)

        print(f"=== 第 {TARGET_MSG} 条 message ===")
        print("subset 属性:", repr(subset_tag))
        print("msg_type 原始:", repr(msg_type_attr), "（callable?", callable(msg_type_attr), "）")

        if msg_type_attr is not None and not callable(msg_type_attr):
            try:
                print("msg_type 作为整数:", int(msg_type_attr))
            except Exception as e:
                print("int(msg_type) 失败:", e)

        if callable(msg_type_attr):
            try:
                val = msg_type_attr()
                print("msg_type() 调用返回:", val)
            except Exception as e:
                print("msg_type() 调用失败:", e)
        found = True
        break

b.close()

if not found:
    print(f"未找到第 {TARGET_MSG} 条 message", file=sys.stderr)
