'''
2025-12-30
This script is to transform the MEI V2 index data into a more convenient format for analysis.
'''
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def parse_mei_file(file_path, output_path='mei_v2.nc'):
    """
    读取MEI文本文件，清洗杂质，并转换为xarray Dataset
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    # 1. 数据清洗
    # 去除 这种标签
    content_clean = re.sub(r'/', ' ', raw_content)
    
    # 去除页脚说明文字 (从 "Multivariate ENSO Index" 开始的内容)
    # 使用 split 丢弃页脚
    content_clean = content_clean.split('Multivariate ENSO Index')[0]

    # 将所有空白字符（换行、tab、空格）替换为单个空格，然后分割
    tokens = content_clean.split()

    # 2. 提取数值流
    # 过滤掉非数字的内容（如果有残留），并转为浮点数
    numeric_data = []
    for t in tokens:
        try:
            numeric_data.append(float(t))
        except ValueError:
            continue

    # 3. 结构化数据
    # 数据流结构通常是: [StartYear, EndYear, Year1, v1..v12, Year2, v1..v12, ...]
    # 我们需要找到由 "年份 + 12个值" 组成的模式
    
    # 这里的 header 往往是 "1979 2025"，这两个数可能会干扰解析，先尝试识别数据块
    # 策略：遍历数组，找到一个整数是年份（1900-2100之间），且后面跟着12个数值
    
    structured_data = []
    years = []
    
    i = 0
    while i < len(numeric_data):
        val = numeric_data[i]
        
        # 判断是否为年份 (整数，且在合理范围内)
        # 注意：数据里的 1979 出现在开头可能是 Header，也可能是第一行数据
        # 真正的行数据特征是：年份后面紧跟的数据通常是很小的浮点数（< 10.0）
        
        is_year = (val.is_integer() and 1950 <= val <= 2050)
        
        if is_year:
            # 检查剩余长度是否足够 (年份 + 12个月)
            if i + 12 < len(numeric_data):
                # 获取接下来12个值
                row_vals = numeric_data[i+1 : i+13]
                
                # 简单的启发式检查：如果接下来12个值里有大整数（比如2025），说明刚才那个不是行头年份
                # 但在这里，MEI指数都是小数值（-3 到 3 左右，或者 -999）
                if all(abs(x) < 10 or x == -999.0 for x in row_vals):
                    years.append(int(val))
                    structured_data.append(row_vals)
                    i += 13 # 跳过这一组
                    continue
        
        # 如果不匹配模式，跳过当前token继续寻找
        i += 1

    if not years:
        raise ValueError("未在文件中解析出有效的数据行，请检查文件格式。")

    # 转为 numpy 数组
    data_matrix = np.array(structured_data)
    
    # 处理缺失值 -999.00
    data_matrix[data_matrix == -999.0] = np.nan

    # 4. 构建 Pandas DataFrame (方便构建时间序列)
    # 根据文件描述：Row values are 2 month seasons (YEAR DJ JF FM MA AM MJ JJ JA AS SO ON ND)
    # 通常我们将这些对应为当年的 1月 到 12月 (或者将DJ视为1月)
    
    # 拉平数据
    flat_data = data_matrix.flatten()
    
    # 构建时间索引
    # 假设第一列 'DJ' 对应的是该年的 1月 (或者该年年初的季节)
    time_index = pd.date_range(start=f'{years[0]}-01-01', periods=len(flat_data), freq='MS')
    
    # 5. 创建 Xarray Dataset
    print(flat_data)
    ds = xr.Dataset(
        data_vars={
            'mei': (('time'), flat_data)
        },
        coords={
            'time': time_index
        },
        attrs={
            'description': 'Multivariate ENSO Index Version 2 (MEI.v2)',
            'source_file': file_path,
            'original_header_info': 'Row values are 2 month seasons (YEAR DJ JF FM MA AM MJ JJ JA AS SO ON ND)',
            'missing_value': -999.00
        }
    )

    # 打印预览
    print("数据处理完成。Dataset 预览：")
    print(ds)
    
    # 保存文件
    if output_path:
        ds.to_netcdf(output_path)
        print(f"\n文件已保存至: {output_path}")
    
    return ds

# --- 执行部分 ---
# 请将 'meiv2 (2).data' 替换为你实际的文件路径
if __name__ == "__main__":
    file_name = '/home/sun/data/download_data/Indexes/meiv2.data' 
    try:
        ds = parse_mei_file(file_name, output_path='mei_v2.nc')
        
        # 可选：如果你想画个图看看
        ds['mei'].plot()
        plt.savefig('/home/sun/paint/donghai/typhoon_prediction/mei_output.png', dpi=300, bbox_inches='tight')
        
    except FileNotFoundError:
        print(f"找不到文件: {file_name}，请确认文件在当前目录下。")