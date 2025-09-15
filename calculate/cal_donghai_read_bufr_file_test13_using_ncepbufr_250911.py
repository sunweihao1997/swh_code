from ncepbufr import open as bufr_open
import numpy as np

def read_wdir_from(bufr_file, target_msg=15658, target_subset=1, print_preview=True):
    """
    读取 BUFR 文件中第 target_msg 条 message 的第 target_subset 个 subset 的 WDIR。
    - 返回: numpy.ndarray，形状为 (n_levels,)；若缺测则用 None 占位。
    - 若该 subset 没有 WDIR，会返回空数组。
    """
    def is_missing(x):
        # BUFR 常见缺测占位是 1e10 或 1e20 等大数；也可能是 NaN
        if x is None:
            return True
        try:
            xf = float(x)
        except Exception:
            return True
        return (not np.isfinite(xf)) or (abs(xf) > 9e9)

    bufr = bufr_open(bufr_file)
    imsg = 0
    got_msg = False
    arr_out = np.array([], dtype=float)

    try:
        # 1) 定位到目标 message
        while bufr.advance() == 0:
            imsg += 1
            if imsg < target_msg:
                continue
            if imsg > target_msg:
                break
            got_msg = True

            nsub = bufr.subsets
            if nsub <= 0:
                if print_preview:
                    print(f"⚠️ 第 {target_msg} 条 message 没有 subset。")
                break
            if target_subset < 1 or target_subset > nsub:
                if print_preview:
                    print(f"⚠️ 第 {target_msg} 条 message 只有 {nsub} 个 subset，"
                          f"请求的 subset={target_subset} 越界。")
                break

            # 2) 在该 message 内定位到目标 subset
            isub = 0
            while bufr.load_subset() == 0:
                isub += 1
                if isub != target_subset:
                    continue

                # 3) 读取该 subset 的 WDIR
                data = bufr.read_subset("WDIR")
                if data is None or getattr(data, "size", 0) == 0:
                    if print_preview:
                        print(f"⚠️ Message #{imsg} Subset #{isub} 中未读取到 WDIR。")
                    break

                # read_subset("WDIR") 通常返回二维 (n_levels, 1)，这里展平成 (n_levels,)
                vals = np.asarray(data, dtype=float).reshape(-1)

                # 将缺测替换为 None，方便查看
                clean = []
                for v in vals:
                    clean.append(None if is_missing(v) else float(v))
                arr_out = np.array(clean, dtype=object)

                if print_preview:
                    print(f"\n✅ Message #{imsg} (subsets={nsub}), Subset #{isub} 的 WDIR：")
                    if arr_out.size == 1:
                        v = arr_out[0]
                        print("  WDIR:", "--" if v is None else v)
                    else:
                        for k, v in enumerate(arr_out, start=1):
                            print(f"  Level {k:02d}: {'--' if v is None else v}")

                break  # 只读这个 subset
            break  # 只读这个 message

        if not got_msg and print_preview:
            print(f"❌ 文件中没有第 {target_msg} 条 message（或尚未到达就结束）。")

    finally:
        bufr.close()

    # 返回 object 数组（包含 None）；如需数值型，可自行筛掉 None 再转换
    return arr_out

# ===== 示例用法 =====
if __name__ == "__main__":
    BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
    # 举例：读取第 15658 条 message 的第 1 个 subset（你可以改成需要的 subset 序号）
    wdir_vals = read_wdir_from(BUFR_FILE, target_msg=15658, target_subset=1, print_preview=True)
