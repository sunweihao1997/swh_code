#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

BUFR_FILE   = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
TARGET_MSG  = 15658
TARGET_SUBSET = 1    # 1-based: 取第 1 个 subset
MISSING_BIG = 9e9

def main():
    try:
        import ncepbufr as bufrmod
    except Exception as e:
        print("请先安装 ncepbufr（pip install ncepbufr 或使用系统包）。")
        raise

    b = bufrmod.open(BUFR_FILE)

    # 可选：设置缺测填充值（ncepbufr 通常返回 masked array，但也支持自定义缺测值）
    try:
        b.set_missing_value(MISSING_BIG)  # 新版接口
    except Exception:
        try:
            b.missing_value = MISSING_BIG  # 某些版本用属性
        except Exception:
            pass

    msg_idx = 0
    found = False

    # 逐条 message 前进
    while b.advance() == 0:  # 返回 0 表示成功推进一条 message
        msg_idx += 1
        if msg_idx == TARGET_MSG:
            found = True

            # 进入该 message 的第 TARGET_SUBSET 个 subset（1-based）
            # 每调用一次 load_subset()，内部指针会移动到下一个 subset
            for _ in range(TARGET_SUBSET):
                rc = b.load_subset()
                if rc != 0:
                    print(f"[错误] 第 {TARGET_MSG} 条 message 不存在第 {TARGET_SUBSET} 个 subset。")
                    b.close()
                    sys.exit(1)

            # 读取 WDIR（可一次读取多个，用空格分隔：'WDIR WSPD'）
            arr = b.read_subset('WDIR')

            # 打印结构信息
            print("=== 返回对象信息（WDIR）===")
            print("类型：", type(arr))
            # ncepbufr 通常返回 numpy.ma.MaskedArray
            try:
                import numpy as np
                print("是否为 masked array：", isinstance(arr, np.ma.MaskedArray))
            except Exception:
                pass

            try:
                print("shape：", getattr(arr, "shape", None))
                print("dtype：", getattr(arr, "dtype", None))
            except Exception:
                pass

            # 打印前几个元素（自动处理 masked）
            try:
                # 展平后看前 10 个
                flat = arr.ravel()
                nshow = min(10, flat.size)
                print(f"前 {nshow} 个值：", flat[:nshow])
                # 如果是 masked，看看有多少缺测
                import numpy as np
                if isinstance(arr, np.ma.MaskedArray):
                    print("缺测个数：", int(arr.mask.sum()) if arr.mask is not np.ma.nomask else 0)
            except Exception as e:
                print("预览值时出错：", e)

            b.close()
            break

    if not found:
        print(f"[错误] 文件中未找到第 {TARGET_MSG} 条 message。")

if __name__ == "__main__":
    main()
