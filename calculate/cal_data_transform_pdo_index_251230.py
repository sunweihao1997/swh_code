'''
2025-12-30
This script is to transform the MEI V2 index data into a more convenient format for analysis.
'''

import pandas as pd
import xarray as xr
import numpy as np
import os

def process_pdo_file(file_path, output_path='/home/sun/data/download_data/Indexes/pdo_ersst_v5.nc'):
    """
    读取ERSST PDO文本文件，清洗并转换为xarray Dataset
    """
    print(f"正在处理文件: {file_path} ...")
    
    try:
        # 1. 读取数据
        # 该文件格式比较规整：空格分隔，前两行是Header
        # 99.99 是常见的缺失值标记，我们在读取时直接处理
        df = pd.read_csv(
            file_path, 
            sep=r'\s+',        # 使用正则表达式匹配任意空白字符作为分隔符
            header=1,          # header在第2行 (索引为1)
            na_values=[99.99, -99.99, 999.9, -999.9], # 定义缺失值列表
            engine='python'    # 使用python引擎更稳定
        )
        
        # 2. 数据清洗
        # 确保 'Year' 列存在并设为索引
        if 'Year' not in df.columns:
            # 尝试修复列名，防止header读取错误
            # 这种文件通常第一列是Year
            df.columns = ['Year'] + list(df.columns[1:])
        
        # 将 Year 设为索引，准备重塑形状
        df = df.set_index('Year')
        
        # 3. 重塑数据 (Wide to Long)
        # 将 (年份, 12个月) 的宽表转换为单列时间序列
        # stack() 会把列名 (Jan, Feb...) 变成二级索引
        series_stacked = df.stack()
        
        # 4. 构建标准时间索引
        # 这里的 series_stacked index 是 MultiIndex (Year, Month_Name)
        # 我们需要将其转换为真正的 datetime 格式
        
        years = series_stacked.index.get_level_values(0)
        # 简单的生成方式：直接根据数据长度生成时间范围
        # 注意：stack() 默认会丢弃 NaN，但我们需要保持时间连续性，所以最好先不 dropna，或者重建索引
        
        # 更稳健的方法：Flatten data values and rebuild time index
        # 确保按年份排序
        df_sorted = df.sort_index()
        raw_values = df_sorted.values.flatten()
        
        # 构建从起始年份 1月 开始的月度索引
        start_year = int(df_sorted.index[0])
        time_index = pd.date_range(
            start=f'{start_year}-01-01', 
            periods=len(raw_values), 
            freq='MS' # Month Start
        )
        
        # 5. 创建 Xarray Dataset
        print(raw_values)
        ds = xr.Dataset(
            data_vars={
                'pdo': (('time'), raw_values)
            },
            coords={
                'time': time_index
            },
            attrs={
                'description': 'ERSST v5 PDO Index',
                'source_file': os.path.basename(file_path),
                'units': 'SST Anomaly (C)',
                'missing_value': np.nan
            }
        )
        
        # 6. 保存
        ds.to_netcdf(output_path)
        print("处理成功！")
        print(ds)
        print(f"\n文件已保存至: {output_path}")
        return ds

    except Exception as e:
        print(f"处理出错: {e}")
        return None

# --- 执行部分 ---
if __name__ == "__main__":
    # 替换为你的文件名
    file_name = '/home/sun/data/download_data/Indexes/ersst.v5.pdo.dat.txt'
    
    if os.path.exists(file_name):
        ds = process_pdo_file(file_name)
        
        # 可选：简单的绘图查看
        if ds:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                ds['pdo'].plot()
                plt.title('PDO Index Time Series')
                plt.savefig('pdo_preview.png')
                print("预览图已保存为 pdo_preview.png")
            except ImportError:
                pass
    else:
        print(f"找不到文件: {file_name}")