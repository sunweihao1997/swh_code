'''
2025-12-30
This script is to transform the IOD index data into a more convenient format for analysis.
'''
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os

def process_iod_file(file_path, output_path='/home/sun/wd_14/data/data/download_data/Indexes/iod_ersst_v5.nc'):
    """
    读取ERSST IOD文本文件 (长格式)，清洗并转换为xarray Dataset
    """
    print(f"正在处理文件: {file_path} ...")
    
    try:
        # 1. 读取数据
        # 文件是以空格分隔的，第一行是表头
        df = pd.read_csv(
            file_path, 
            sep=r'\s+',        # 匹配任意空白字符
            engine='python'
        )
        
        # 2. 构建时间索引
        # 将 Year 和 month 列合并转换为 datetime 对象
        # 格式: YYYY-MM-01
        df['time'] = pd.to_datetime(
            df['Year'].astype(str) + '-' + df['month'].astype(str) + '-01'
        )
        
        # 设置时间为索引
        df = df.set_index('time')
        
        # 3. 提取 IOD 指数 (Diff 列)
        # 根据文件头: Year month West EAST Diff
        # 通常 Diff 就是 IOD Index
        target_col = 'Diff' 
        if target_col not in df.columns:
            raise ValueError(f"未在文件中找到 '{target_col}' 列，现有列名: {df.columns}")
            
        iod_series = df[target_col]
        
        # 4. 创建 Xarray Dataset
        
        ds = xr.Dataset(
            data_vars={
                'iod': (('time'), iod_series.values)
            },
            coords={
                'time': iod_series.index
            },
            attrs={
                'description': 'ERSST v5 IOD Index (Dipole Mode Index)',
                'source_file': os.path.basename(file_path),
                'units': 'SST Anomaly (C)',
                'columns_extracted': 'Diff (West - East)'
            }
        )
        print(ds['iod'])
        
        # 5. 保存 NetCDF
        ds.to_netcdf(output_path)
        print("数据处理成功！")
        print(ds)
        print(f"\n文件已保存至: {output_path}")
        
        return ds

    except Exception as e:
        print(f"处理出错: {e}")
        return None

# --- 执行部分 ---
if __name__ == "__main__":
    file_name = '/home/sun/wd_14/data/data/download_data/Indexes/ersst.v5.iod.dat.txt'
    
    if os.path.exists(file_name):
        ds = process_iod_file(file_name)
        
        # 绘图并保存
        if ds:
            plt.figure(figsize=(12, 6))
            ds['iod'].plot()
            plt.title('Indian Ocean Dipole (IOD) Index Time Series')
            plt.ylabel('IOD Index (Diff)')
            plt.grid(True, alpha=0.3)
            
            # 保存图片
            img_name = '/home/sun/wd_14/data/data/download_data/Indexes/iod_timeseries.png'
            plt.savefig(img_name, dpi=300, bbox_inches='tight')
            print(f"预览图已保存为 {img_name}")
            
    else:
        print(f"找不到文件: {file_name}")