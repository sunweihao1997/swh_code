from pathlib import Path
import re
import pandas as pd


def clean_cn_id(cn_id_raw: str):
    """
    清洗我国编号：
    - 正常: 1801 -> 1801
    - 异常: 7127,7128 -> 7127
    返回:
    - cn_id_clean
    - issue_msg (没有问题则为 None)
    """
    cn_id_raw = str(cn_id_raw).strip()

    if "," in cn_id_raw:
        parts = [x.strip() for x in cn_id_raw.split(",") if x.strip()]
        cn_id_clean = parts[0] if parts else cn_id_raw
        issue_msg = f"cn_id 多编号: 原始='{cn_id_raw}' -> 保留第一个='{cn_id_clean}'"
        return cn_id_clean, issue_msg

    return cn_id_raw, None


def clean_en_name(en_name_raw: str):
    """
    清洗英文名：
    - 正常: BOLAVEN -> BOLAVEN
    - 异常: Faye(Gloria) -> Faye
    返回:
    - en_name_clean
    - issue_msg (没有问题则为 None)
    """
    en_name_raw = str(en_name_raw).strip()

    # 去掉括号及其内容
    en_name_clean = re.sub(r"\(.*?\)", "", en_name_raw).strip()

    if en_name_clean != en_name_raw:
        issue_msg = f"en_name 含括号别名: 原始='{en_name_raw}' -> 清洗后='{en_name_clean}'"
        return en_name_clean, issue_msg

    return en_name_raw, None


def parse_cma_best_track(file_path: str):
    """
    解析单个 CMA 台风最佳路径 TXT（如 CH2018BST.txt）

    返回：
    - storms_df: 每个台风一行（头记录）
    - tracks_df: 每个时次一行（路径记录）
    - issues_df: 该文件中发现并清洗过的异常记录日志
    """

    file_path = Path(file_path)

    header_pattern = re.compile(
        r'^(\d{5})\s+'      # 66666
        r'(\d{4})\s+'       # 国际编号
        r'(\d+)\s+'         # 路径记录行数
        r'(\d{4})\s+'       # 热带气旋序号（含热带低压）
        r'([^\s]+)\s+'      # 我国编号，允许 1801 或 7127,7128
        r'(\d)\s+'          # 终结标记
        r'(\d)\s+'          # 路径时间间隔小时数
        r'(.+?)\s+'         # 英文名
        r'(\d{8})$'         # 数据集形成日期
    )

    storms = []
    tracks = []
    issues = []

    current_storm = None
    current_record_no = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            # 1) 头记录
            if line.startswith("66666"):
                m = header_pattern.match(line)
                if not m:
                    raise ValueError(f"第 {line_no} 行头记录解析失败：{line}")

                (
                    rec_flag,
                    intl_id,
                    record_count,
                    tc_seq,
                    cn_id_raw,
                    end_flag,
                    interval_hours,
                    en_name_raw,
                    dataset_date,
                ) = m.groups()

                cn_id_clean, cn_id_issue = clean_cn_id(cn_id_raw)
                en_name_clean, en_name_issue = clean_en_name(en_name_raw)

                storm_index = len(storms) + 1

                if cn_id_issue is not None:
                    issues.append({
                        "file_name": file_path.name,
                        "line_no": line_no,
                        "storm_index": storm_index,
                        "issue_type": "cn_id_multi",
                        "raw_value": cn_id_raw,
                        "clean_value": cn_id_clean,
                        "message": cn_id_issue,
                        "raw_line": line,
                    })

                if en_name_issue is not None:
                    issues.append({
                        "file_name": file_path.name,
                        "line_no": line_no,
                        "storm_index": storm_index,
                        "issue_type": "en_name_alias",
                        "raw_value": en_name_raw,
                        "clean_value": en_name_clean,
                        "message": en_name_issue,
                        "raw_line": line,
                    })

                current_storm = {
                    "storm_index": storm_index,
                    "record_flag": rec_flag,
                    "intl_id": intl_id,
                    "record_count": int(record_count),
                    "tc_seq": tc_seq,

                    # 保留原始字段
                    "cn_id_raw": cn_id_raw,
                    "en_name_raw": en_name_raw.strip(),

                    # 使用清洗后的字段
                    "cn_id": cn_id_clean,
                    "en_name": en_name_clean,

                    "end_flag": int(end_flag),
                    "interval_hours": int(interval_hours),
                    "dataset_date": pd.to_datetime(dataset_date, format="%Y%m%d"),
                }
                storms.append(current_storm)
                current_record_no = 0
                continue

            # 2) 路径记录
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
                "cn_id_raw": current_storm["cn_id_raw"],
                "en_name": current_storm["en_name"],
                "en_name_raw": current_storm["en_name_raw"],
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
    issues_df = pd.DataFrame(issues)

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

    # 过程最大级别
    max_grade = (
        tracks_df.groupby("storm_index")["grade_code"]
        .max()
        .rename("max_grade_code")
    )
    storms_df = storms_df.merge(max_grade, on="storm_index", how="left")
    storms_df["max_grade_name"] = storms_df["max_grade_code"].map(grade_map)

    # 校验记录数
    actual_counts = tracks_df.groupby("storm_index").size().rename("actual_count")
    storms_df = storms_df.merge(actual_counts, on="storm_index", how="left")
    storms_df["count_match"] = storms_df["record_count"] == storms_df["actual_count"]

    return storms_df, tracks_df, issues_df


def batch_process_cma_best_track(input_dir: str, output_dir: str):
    """
    批量处理目录下所有 CH????BST.txt 文件
    1. 每年分别保存 storms / tracks
    2. 合并所有年份，保存总表
    3. 输出异常清洗日志
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    yearly_storms_dir = output_dir / "yearly_storms"
    yearly_tracks_dir = output_dir / "yearly_tracks"
    yearly_logs_dir = output_dir / "yearly_logs"

    yearly_storms_dir.mkdir(parents=True, exist_ok=True)
    yearly_tracks_dir.mkdir(parents=True, exist_ok=True)
    yearly_logs_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("CH????BST.txt"))
    if not txt_files:
        raise FileNotFoundError(f"目录下未找到 CH????BST.txt 文件：{input_dir}")

    all_storms = []
    all_tracks = []
    all_issues = []
    failed_files = []

    for txt_file in txt_files:
        m = re.match(r"CH(\d{4})BST\.txt$", txt_file.name)
        if not m:
            print(f"跳过不符合命名规则的文件：{txt_file.name}")
            continue

        year = int(m.group(1))
        print(f"正在处理 {txt_file.name} ...")

        try:
            storms_df, tracks_df, issues_df = parse_cma_best_track(txt_file)
        except Exception as e:
            print(f"处理失败 {txt_file.name}: {e}")
            failed_files.append({"file_name": txt_file.name, "error": str(e)})
            continue

        storms_df["year"] = year
        tracks_df["year"] = year
        if not issues_df.empty:
            issues_df["year"] = year

        # 全局唯一键
        storms_df["storm_uid"] = storms_df.apply(
            lambda row: f"{row['year']}_{int(row['storm_index']):04d}", axis=1
        )

        uid_map = storms_df[["storm_index", "storm_uid"]].copy()
        tracks_df = tracks_df.merge(uid_map, on="storm_index", how="left")

        if not issues_df.empty:
            issues_df = issues_df.merge(uid_map, on="storm_index", how="left")

        # 列顺序
        storms_cols = [
            "storm_uid", "year", "storm_index",
            "record_flag", "intl_id", "record_count",
            "tc_seq",
            "cn_id", "cn_id_raw",
            "end_flag", "interval_hours",
            "en_name", "en_name_raw",
            "dataset_date",
            "max_grade_code", "max_grade_name",
            "actual_count", "count_match"
        ]
        storms_df = storms_df[storms_cols]

        tracks_cols = [
            "storm_uid", "year", "storm_index", "record_no",
            "intl_id", "tc_seq",
            "cn_id", "cn_id_raw",
            "en_name", "en_name_raw",
            "obs_time_utc", "grade_code", "grade_name",
            "lat", "lon", "pressure_hpa", "wind_ms", "owd_ms"
        ]
        tracks_df = tracks_df[tracks_cols]

        # 每年输出
        storms_outfile = yearly_storms_dir / f"storms_{year}.csv"
        tracks_outfile = yearly_tracks_dir / f"tracks_{year}.csv"
        log_outfile = yearly_logs_dir / f"issues_{year}.csv"

        storms_df.to_csv(storms_outfile, index=False, encoding="utf-8-sig")
        tracks_df.to_csv(tracks_outfile, index=False, encoding="utf-8-sig")

        if not issues_df.empty:
            issues_df.to_csv(log_outfile, index=False, encoding="utf-8-sig")

        # 终端日志摘要
        if issues_df.empty:
            print(f"  {year}: 未发现需清洗的异常头记录")
        else:
            issue_count = len(issues_df)
            issue_storms = issues_df["storm_index"].nunique()
            print(f"  {year}: 发现并清洗 {issue_count} 条异常记录，涉及 {issue_storms} 个台风，已写入 {log_outfile.name}")

        all_storms.append(storms_df)
        all_tracks.append(tracks_df)
        if not issues_df.empty:
            all_issues.append(issues_df)

    if not all_storms or not all_tracks:
        raise RuntimeError("没有成功处理任何文件，请检查输入数据或日志。")

    # 合并所有年份
    all_storms_df = pd.concat(all_storms, ignore_index=True)
    all_tracks_df = pd.concat(all_tracks, ignore_index=True)

    all_storms_df.to_csv(output_dir / "all_storms.csv", index=False, encoding="utf-8-sig")
    all_tracks_df.to_csv(output_dir / "all_tracks.csv", index=False, encoding="utf-8-sig")

    merged_all_df = all_tracks_df.merge(
        all_storms_df[
            [
                "storm_uid", "record_count", "end_flag", "interval_hours", "dataset_date",
                "max_grade_code", "max_grade_name", "count_match"
            ]
        ],
        on="storm_uid",
        how="left",
    )
    merged_all_df.to_csv(output_dir / "all_tracks_merged.csv", index=False, encoding="utf-8-sig")

    # 总日志
    if all_issues:
        all_issues_df = pd.concat(all_issues, ignore_index=True)
    else:
        all_issues_df = pd.DataFrame(columns=[
            "file_name", "line_no", "storm_index", "issue_type",
            "raw_value", "clean_value", "message", "raw_line", "year", "storm_uid"
        ])

    all_issues_df.to_csv(output_dir / "all_issues.csv", index=False, encoding="utf-8-sig")

    if failed_files:
        failed_df = pd.DataFrame(failed_files)
        failed_df.to_csv(output_dir / "failed_files.csv", index=False, encoding="utf-8-sig")
        print(f"\n有 {len(failed_files)} 个文件处理失败，详见 failed_files.csv")
    else:
        print("\n所有文件处理完成，无失败文件。")

    print(f"总台风数：{len(all_storms_df)}")
    print(f"总路径记录数：{len(all_tracks_df)}")
    print(f"异常清洗记录数：{len(all_issues_df)}")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    input_dir = "/home/sun/data/download_data/CMA_typhoon/CMABSTdata"
    output_dir = "/home/sun/data/download_data/CMA_typhoon/CMABSTdata_output"

    batch_process_cma_best_track(input_dir, output_dir)