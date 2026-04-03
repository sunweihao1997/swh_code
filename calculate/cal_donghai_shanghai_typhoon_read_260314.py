from pathlib import Path
import re
import pandas as pd


from pathlib import Path
import re
import pandas as pd


def parse_cma_best_track(file_path: str):
    """
    解析 CMA 台风最佳路径 TXT（如 CH2018BST.txt）

    返回：
    - storms_df: 每个台风一行（头记录）
    - tracks_df: 每个时次一行（路径记录）
    """

    file_path = Path(file_path)

    header_pattern = re.compile(
        r'^(\d{5})\s+'
        r'(\d{4})\s+'
        r'(\d+)\s+'
        r'(\d{4})\s+'
        r'(\d{4})\s+'
        r'(\d)\s+'
        r'(\d)\s+'
        r'(.+?)\s+'
        r'(\d{8})$'
    )

    storms = []
    tracks = []
    current_storm = None
    current_record_no = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            if line.startswith("66666"):
                m = header_pattern.match(line)
                if not m:
                    raise ValueError(f"第 {line_no} 行头记录解析失败：{line}")

                (
                    rec_flag,
                    intl_id,
                    record_count,
                    tc_seq,
                    cn_id,
                    end_flag,
                    interval_hours,
                    en_name,
                    dataset_date,
                ) = m.groups()

                current_storm = {
                    "storm_index": len(storms) + 1,
                    "record_flag": rec_flag,
                    "intl_id": intl_id,
                    "record_count": int(record_count),
                    "tc_seq": tc_seq,
                    "cn_id": cn_id,
                    "end_flag": int(end_flag),
                    "interval_hours": int(interval_hours),
                    "en_name": en_name.strip(),
                    "dataset_date": pd.to_datetime(dataset_date, format="%Y%m%d"),
                }
                storms.append(current_storm)
                current_record_no = 0
                continue

            if current_storm is None:
                raise ValueError(f"第 {line_no} 行在第一个头记录之前出现路径记录：{line}")

            parts = line.split()

            if len(parts) == 6:
                obs_time, grade, lat10, lon10, pres, wnd = parts
                owd = None
            elif len(parts) == 7:
                obs_time, grade, lat10, lon10, pres, wnd, owd = parts
            else:
                raise ValueError(
                    f"第 {line_no} 行路径记录列数异常（{len(parts)} 列）：{line}"
                )

            current_record_no += 1

            tracks.append({
                "storm_index": current_storm["storm_index"],
                "record_no": current_record_no,
                "intl_id": current_storm["intl_id"],
                "tc_seq": current_storm["tc_seq"],
                "cn_id": current_storm["cn_id"],
                "en_name": current_storm["en_name"],
                "obs_time_utc": pd.to_datetime(obs_time, format="%Y%m%d%H"),
                "grade_code": int(grade),
                "lat": int(lat10) / 10.0,
                "lon": int(lon10) / 10.0,
                "pressure_hpa": int(pres),
                "wind_ms": int(wnd),
                "owd_ms": int(owd) if owd is not None else pd.NA,
            })

    storms_df = pd.DataFrame(storms)
    tracks_df = pd.DataFrame(tracks)

    grade_map = {
        0: "弱于热带低压/未知",
        1: "热带低压",
        2: "热带风暴",
        3: "强热带风暴",
        4: "台风",
        5: "强台风",
        6: "超强台风",
        9: "变性",
    }
    tracks_df["grade_name"] = tracks_df["grade_code"].map(grade_map)

    # 新增：过程最大级别
    max_grade = (
        tracks_df.groupby("storm_index")["grade_code"]
        .max()
        .rename("max_grade_code")
    )
    print(max_grade.head())
    storms_df = storms_df.merge(max_grade, on="storm_index", how="left")
    storms_df["max_grade_name"] = storms_df["max_grade_code"].map(grade_map)

    # 校验记录数
    actual_counts = tracks_df.groupby("storm_index").size().rename("actual_count")
    storms_df = storms_df.merge(actual_counts, on="storm_index", how="left")
    storms_df["count_match"] = storms_df["record_count"] == storms_df["actual_count"]

    return storms_df, tracks_df


if __name__ == "__main__":
    txt_path = "/home/sun/data/download_data/CMA_typhoon/CMABSTdata/"   # 改成你的文件路径
    txt_path = txt_path + "CH2018BST.txt"  # 2018年台风最佳路径数据文件
    storms_df, tracks_df = parse_cma_best_track(txt_path)

#    print("台风个数：", len(storms_df))
#    print("路径记录数：", len(tracks_df))
#    print("\n头记录前5行：")
    print(storms_df.head())
#    print("\n路径记录前5行：")
    print(tracks_df.head())

    # 导出
    storms_df.to_csv("storms_2018.csv", index=False, encoding="utf-8-sig")
    tracks_df.to_csv("tracks_2018.csv", index=False, encoding="utf-8-sig")

    # 如果你想要一个“扁平化”的总表，也可以合并
    merged_df = tracks_df.merge(
        storms_df[
            ["storm_index", "record_count", "end_flag", "interval_hours", "dataset_date"]
        ],
        on="storm_index",
        how="left",
    )
#    merged_df.to_csv("tracks_2018_merged.csv", index=False, encoding="utf-8-sig")
#    print(merged_df.head())

#    print("\n已输出：storms_2018.csv / tracks_2018.csv / tracks_2018_merged.csv")