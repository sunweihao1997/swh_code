'''
2025-12-28
This script serves as a response for the ISO-monsoon onset article.
Calculate and plot the respective land and ocean PSL anomaly in monsoon onset early and late years.
'''
import numpy as np
import matplotlib.pyplot as plt

# ============== 配置参数 ==============
USE_REAL_DATA = True  # 设为True使用真实数据，False使用模拟数据

# 文件路径（使用真实数据时需要修改）
PSL_FILE = '/home/sun/wd_14/data_beijing/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_OLR.nc'

# ============== 定义早年和晚年 ==============
# 根据用户提供的信息
year_early = [1984, 1985, 1999, 2000, 2009, 2017,]
year_late = [1983, 1987, 1993, 1997, 2010, 2021, 2016, 2018,]

# 数据起止年份
start_year = 1980
end_year = 2021  # 修改：数据实际到2021年，504个月 = 42年
n_years = end_year - start_year + 1  # 42年

# ============== 数据处理函数 ==============
def load_real_data(filepath):
    """加载真实的netCDF数据"""
    import xarray as xr
    ds = xr.open_dataset(filepath)
    psl_land = ds['LSTC_psl_7090'].values  # 海洋 psl (7090区域)
    psl_ocean = ds['LSTC_psl_IOB'].values    # 陆地/IOB区域 psl
    ds.close()
    return psl_ocean, psl_land

def generate_mock_data():
    """生成模拟数据用于演示"""
    np.random.seed(42)
    n_months = n_years * 12
    
    # 创建有季节变化的模拟数据
    # 海洋psl - 基础值约101325 Pa，加上季节变化和随机扰动
    base_ocean = 101325
    seasonal_ocean = np.tile(np.array([200, 150, 100, 50, 0, -50, -100, -50, 0, 50, 100, 150]), n_years)
    noise_ocean = np.random.normal(0, 30, n_months)
    psl_ocean = base_ocean + seasonal_ocean + noise_ocean
    
    # 陆地psl - 变化幅度更大
    base_land = 101000
    seasonal_land = np.tile(np.array([300, 200, 100, -50, -150, -200, -150, -100, 0, 100, 200, 250]), n_years)
    noise_land = np.random.normal(0, 50, n_months)
    psl_land = base_land + seasonal_land + noise_land
    
    # 为早年和晚年添加系统性偏差
    for yr in year_early:
        if start_year <= yr <= end_year:
            idx = (yr - start_year) * 12
            # 早年：3-6月海洋psl偏高，陆地psl偏低（有利于早爆发）
            psl_ocean[idx+2:idx+6] += np.array([80, 100, 90, 60])
            psl_land[idx+2:idx+6] -= np.array([60, 80, 100, 80])
    
    for yr in year_late:
        if start_year <= yr <= end_year:
            idx = (yr - start_year) * 12
            # 晚年：3-6月海洋psl偏低，陆地psl偏高（不利于早爆发）
            psl_ocean[idx+2:idx+6] -= np.array([70, 90, 80, 50])
            psl_land[idx+2:idx+6] += np.array([50, 70, 90, 70])
    
    return psl_ocean, psl_land

def compute_monthly_climatology(data, n_years):
    """计算月气候态"""
    monthly_clim = np.zeros(12)
    data_reshaped = data.reshape(n_years, 12)
    for m in range(12):
        monthly_clim[m] = np.nanmean(data_reshaped[:, m])
    return monthly_clim

def compute_composite(years_list, data, start_year, end_year, months_indices):
    """
    计算指定年份列表的月份合成值
    months_indices: 月份索引列表，如[2,3,4,5]代表3-6月
    """
    composites = {i: [] for i in months_indices}
    
    for yr in years_list:
        if start_year <= yr <= end_year:
            yr_idx = yr - start_year
            for m in months_indices:
                idx = yr_idx * 12 + m
                if idx < len(data):
                    composites[m].append(data[idx])
    
    result = {}
    month_names = {2: 'Mar', 3: 'Apr', 4: 'May', 5: 'Jun'}
    for m in months_indices:
        result[month_names[m]] = np.nanmean(composites[m]) if composites[m] else np.nan
    
    return result

def add_value_labels(ax, bars, fontsize=9):
    """在柱状图上添加数值标签"""
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, offset),
                       textcoords="offset points",
                       ha='center', va=va, fontsize=fontsize)

# ============== 主程序 ==============
print("=" * 60)
print("季风爆发早年/晚年 海平面气压异常分析")
print("=" * 60)

# 加载数据
if USE_REAL_DATA:
    print("\n正在加载真实数据...")
    try:
        psl_ocean, psl_land = load_real_data(PSL_FILE)
        # 自动计算n_years
        actual_months = len(psl_ocean)
        actual_years = actual_months // 12
        print(f"成功加载数据，共 {actual_months} 个月 ({actual_years} 年)")
        
        # 如果数据长度与设定不符，自动调整
        if actual_years != n_years:
            print(f"注意：数据实际年数({actual_years})与设定({n_years})不符，自动调整...")
            n_years = actual_years
            end_year = start_year + n_years - 1
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {PSL_FILE}")
        print("将使用模拟数据继续...")
        psl_ocean, psl_land = generate_mock_data()
else:
    print("\n使用模拟数据进行演示...")
    psl_ocean, psl_land = generate_mock_data()

# 计算气候态
print(f"\n正在计算气候态（{start_year}-{end_year}）...")
clim_ocean = compute_monthly_climatology(psl_ocean, n_years)
clim_land = compute_monthly_climatology(psl_land, n_years)

# 计算早年和晚年的3-6月合成
print("正在计算早年合成...")
months_idx = [2, 3, 4, 5]  # 3月=索引2, 4月=索引3, 5月=索引4, 6月=索引5
early_ocean = compute_composite(year_early, psl_ocean, start_year, end_year, months_idx)
early_land = compute_composite(year_early, psl_land, start_year, end_year, months_idx)

print("正在计算晚年合成...")
late_ocean = compute_composite(year_late, psl_ocean, start_year, end_year, months_idx)
late_land = compute_composite(year_late, psl_land, start_year, end_year, months_idx)

# 计算异常（合成 - 气候态）
month_names = ['Mar', 'Apr', 'May', 'Jun']
clim_indices = [2, 3, 4, 5]

early_ocean_anom = {m: early_ocean[m] - clim_ocean[i] for m, i in zip(month_names, clim_indices)}
early_land_anom = {m: early_land[m] - clim_land[i] for m, i in zip(month_names, clim_indices)}
late_ocean_anom = {m: late_ocean[m] - clim_ocean[i] for m, i in zip(month_names, clim_indices)}
late_land_anom = {m: late_land[m] - clim_land[i] for m, i in zip(month_names, clim_indices)}

# 打印结果
print("\n" + "=" * 60)
print("结果汇总")
print("=" * 60)

valid_early_years = [y for y in year_early if start_year <= y <= end_year]
valid_late_years = [y for y in year_late if start_year <= y <= end_year]

print(f"\n早年 (Early Onset Years): {valid_early_years}")
print("-" * 40)
print(f"{'月份':<8}{'海洋异常(Pa)':<15}{'陆地异常(Pa)':<15}")
for m in month_names:
    print(f"{m:<8}{early_ocean_anom[m]:<15.2f}{early_land_anom[m]:<15.2f}")

print(f"\n晚年 (Late Onset Years): {valid_late_years}")
print("-" * 40)
print(f"{'月份':<8}{'海洋异常(Pa)':<15}{'陆地异常(Pa)':<15}")
for m in month_names:
    print(f"{m:<8}{late_ocean_anom[m]:<15.2f}{late_land_anom[m]:<15.2f}")

# ============== 绘图 ==============
print("\n正在生成图表...")

# 图1：分别展示早年和晚年（包含差值）
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(month_names))
width = 0.25  # 调整宽度以容纳3个柱子

ocean_color = '#1f77b4'  # 蓝色 - 海洋
land_color = '#d62728'   # 红色 - 陆地
diff_color = '#2ecc71'   # 绿色 - 差值

# 左图：早年
ax1 = axes1[0]
ocean_vals_early = [early_ocean_anom[m] for m in month_names]
land_vals_early = [early_land_anom[m] for m in month_names]
diff_vals_early = [early_land_anom[m] - early_ocean_anom[m] for m in month_names]  # Land - Ocean

# 顺序：Land -> Ocean -> Difference
bars1 = ax1.bar(x - width, land_vals_early, width, label='Land (IOB)', color=land_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x, ocean_vals_early, width, label='Ocean (7090)', color=ocean_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_diff1 = ax1.bar(x + width, diff_vals_early, width, label='Land - Ocean', color=diff_color, alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
ax1.set_ylabel('PSL Anomaly (Pa)', fontsize=12, fontweight='bold')
ax1.set_title('Early Monsoon Onset Years\n(Composite - Climatology)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(month_names, fontsize=11)
ax1.legend(loc='best', fontsize=10, framealpha=0.9)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_axisbelow(True)
add_value_labels(ax1, bars1)
add_value_labels(ax1, bars2)
add_value_labels(ax1, bars_diff1)

# 右图：晚年
ax2 = axes1[1]
ocean_vals_late = [late_ocean_anom[m] for m in month_names]
land_vals_late = [late_land_anom[m] for m in month_names]
diff_vals_late = [late_land_anom[m] - late_ocean_anom[m] for m in month_names]  # Land - Ocean

# 顺序：Land -> Ocean -> Difference
bars3 = ax2.bar(x - width, land_vals_late, width, label='Land (IOB)', color=land_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars4 = ax2.bar(x, ocean_vals_late, width, label='Ocean (7090)', color=ocean_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_diff2 = ax2.bar(x + width, diff_vals_late, width, label='Land - Ocean', color=diff_color, alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
ax2.set_ylabel('PSL Anomaly (Pa)', fontsize=12, fontweight='bold')
ax2.set_title('Late Monsoon Onset Years\n(Composite - Climatology)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(month_names, fontsize=11)
ax2.legend(loc='best', fontsize=10, framealpha=0.9)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_axisbelow(True)
add_value_labels(ax2, bars3)
add_value_labels(ax2, bars4)
add_value_labels(ax2, bars_diff2)

# 统一Y轴范围
all_vals = ocean_vals_early + land_vals_early + diff_vals_early + ocean_vals_late + land_vals_late + diff_vals_late
all_vals = [v for v in all_vals if not np.isnan(v)]
if all_vals:
    margin = max(abs(min(all_vals)), abs(max(all_vals))) * 0.3
    y_limit = max(abs(min(all_vals)), abs(max(all_vals))) + margin
    ax1.set_ylim(-y_limit, y_limit)
    ax2.set_ylim(-y_limit, y_limit)

plt.tight_layout()
output1 = '/home/sun/paint/CD/response/monsoon_psl_anomaly_by_category.png'
plt.savefig(output1, dpi=150, bbox_inches='tight', facecolor='white')
print(f"图1已保存: {output1}")
plt.close()

# 图2：早年vs晚年对比（包含差值对比）
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

early_color = '#27ae60'  # 绿色 - 早年
late_color = '#e74c3c'   # 红色 - 晚年
width = 0.35

# 左图：陆地psl
ax3 = axes2[0]
bars5 = ax3.bar(x - width/2, land_vals_early, width, label='Early Onset', color=early_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars6 = ax3.bar(x + width/2, land_vals_late, width, label='Late Onset', color=late_color, alpha=0.8, edgecolor='black', linewidth=0.5)

ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
ax3.set_ylabel('PSL Anomaly (Pa)', fontsize=12, fontweight='bold')
ax3.set_title('Land (IOB) PSL Anomaly\nEarly vs Late Onset', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(month_names, fontsize=11)
ax3.legend(loc='best', fontsize=10, framealpha=0.9)
ax3.grid(axis='y', linestyle='--', alpha=0.5)
ax3.set_axisbelow(True)
add_value_labels(ax3, bars5)
add_value_labels(ax3, bars6)

# 中图：海洋psl
ax4 = axes2[1]
bars7 = ax4.bar(x - width/2, ocean_vals_early, width, label='Early Onset', color=early_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars8 = ax4.bar(x + width/2, ocean_vals_late, width, label='Late Onset', color=late_color, alpha=0.8, edgecolor='black', linewidth=0.5)

ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
ax4.set_ylabel('PSL Anomaly (Pa)', fontsize=12, fontweight='bold')
ax4.set_title('Ocean (7090) PSL Anomaly\nEarly vs Late Onset', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(month_names, fontsize=11)
ax4.legend(loc='best', fontsize=10, framealpha=0.9)
ax4.grid(axis='y', linestyle='--', alpha=0.5)
ax4.set_axisbelow(True)
add_value_labels(ax4, bars7)
add_value_labels(ax4, bars8)

# 右图：Land-Ocean差值对比
ax5 = axes2[2]
bars9 = ax5.bar(x - width/2, diff_vals_early, width, label='Early Onset', color=early_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars10 = ax5.bar(x + width/2, diff_vals_late, width, label='Late Onset', color=late_color, alpha=0.8, edgecolor='black', linewidth=0.5)

ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax5.set_xlabel('Month', fontsize=12, fontweight='bold')
ax5.set_ylabel('PSL Anomaly Difference (Pa)', fontsize=12, fontweight='bold')
ax5.set_title('Land - Ocean PSL Anomaly\nEarly vs Late Onset', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(month_names, fontsize=11)
ax5.legend(loc='best', fontsize=10, framealpha=0.9)
ax5.grid(axis='y', linestyle='--', alpha=0.5)
ax5.set_axisbelow(True)
add_value_labels(ax5, bars9)
add_value_labels(ax5, bars10)

# 统一Y轴范围
if all_vals:
    ax3.set_ylim(-y_limit, y_limit)
    ax4.set_ylim(-y_limit, y_limit)
    ax5.set_ylim(-y_limit, y_limit)

plt.tight_layout()
output2 = '/home/sun/paint/CD/response/monsoon_psl_early_vs_late.png'
plt.savefig(output2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"图2已保存: {output2}")
plt.close()

# 图3：六面板综合图（包含差值）
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
width = 0.25

# 子图1：早年海洋+陆地+差值
ax = axes3[0, 0]
# 顺序：Land -> Ocean -> Difference
bars_a = ax.bar(x - width, land_vals_early, width, label='Land', color=land_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_b = ax.bar(x, ocean_vals_early, width, label='Ocean', color=ocean_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_c = ax.bar(x + width, diff_vals_early, width, label='Land-Ocean', color=diff_color, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('PSL Anomaly (Pa)', fontsize=11, fontweight='bold')
ax.set_title('(a) Early Onset: Land, Ocean & Difference', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(month_names)
ax.legend(loc='best', fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 子图2：晚年海洋+陆地+差值
ax = axes3[0, 1]
# 顺序：Land -> Ocean -> Difference
bars_d = ax.bar(x - width, land_vals_late, width, label='Land', color=land_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_e = ax.bar(x, ocean_vals_late, width, label='Ocean', color=ocean_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_f = ax.bar(x + width, diff_vals_late, width, label='Land-Ocean', color=diff_color, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('PSL Anomaly (Pa)', fontsize=11, fontweight='bold')
ax.set_title('(b) Late Onset: Land, Ocean & Difference', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(month_names)
ax.legend(loc='best', fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 子图3：差值 早年vs晚年
ax = axes3[0, 2]
width_comp = 0.35
bars_g = ax.bar(x - width_comp/2, diff_vals_early, width_comp, label='Early', color=early_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_h = ax.bar(x + width_comp/2, diff_vals_late, width_comp, label='Late', color=late_color, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('PSL Difference (Pa)', fontsize=11, fontweight='bold')
ax.set_title('(c) Land-Ocean: Early vs Late', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(month_names)
ax.legend(loc='best', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 子图4：陆地 早年vs晚年
ax = axes3[1, 0]
bars_i = ax.bar(x - width_comp/2, land_vals_early, width_comp, label='Early', color=early_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_j = ax.bar(x + width_comp/2, land_vals_late, width_comp, label='Late', color=late_color, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Month', fontsize=11, fontweight='bold')
ax.set_ylabel('PSL Anomaly (Pa)', fontsize=11, fontweight='bold')
ax.set_title('(d) Land: Early vs Late Onset', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(month_names)
ax.legend(loc='best', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 子图5：海洋 早年vs晚年
ax = axes3[1, 1]
bars_k = ax.bar(x - width_comp/2, ocean_vals_early, width_comp, label='Early', color=early_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_l = ax.bar(x + width_comp/2, ocean_vals_late, width_comp, label='Late', color=late_color, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Month', fontsize=11, fontweight='bold')
ax.set_ylabel('PSL Anomaly (Pa)', fontsize=11, fontweight='bold')
ax.set_title('(e) Ocean: Early vs Late Onset', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(month_names)
ax.legend(loc='best', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 子图6：早年-晚年的差异
ax = axes3[1, 2]
ocean_diff = [ocean_vals_early[i] - ocean_vals_late[i] for i in range(len(month_names))]
land_diff = [land_vals_early[i] - land_vals_late[i] for i in range(len(month_names))]
landocean_diff = [diff_vals_early[i] - diff_vals_late[i] for i in range(len(month_names))]

# 顺序：Land -> Ocean -> Difference
bars_m = ax.bar(x - width, land_diff, width, label='Land', color=land_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_n = ax.bar(x, ocean_diff, width, label='Ocean', color=ocean_color, alpha=0.8, edgecolor='black', linewidth=0.5)
bars_o = ax.bar(x + width, landocean_diff, width, label='Land-Ocean', color=diff_color, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Month', fontsize=11, fontweight='bold')
ax.set_ylabel('PSL Difference (Pa)', fontsize=11, fontweight='bold')
ax.set_title('(f) Early minus Late Onset', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(month_names)
ax.legend(loc='best', fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 统一所有子图Y轴范围
for ax in axes3.flat:
    ax.set_ylim(-y_limit, y_limit)

fig3.suptitle('Monsoon Onset PSL Anomaly Analysis (Mar-Jun)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
output3 = '/home/sun/paint/CD/response/monsoon_psl_comprehensive.png'
plt.savefig(output3, dpi=150, bbox_inches='tight', facecolor='white')
print(f"图3已保存: {output3}")
plt.close()

print("\n" + "=" * 60)
print("程序运行完成！")
print("=" * 60)
print("\n生成的图表文件：")
print(f"  1. {output1}")
print(f"  2. {output2}")
print(f"  3. {output3}")

if not USE_REAL_DATA:
    print("\n注意：当前使用的是模拟数据。")
    print("如需使用真实数据，请：")
    print("  1. 上传ERA5_LSTC_1980_2021_t2m_ts_sp_psl_OLR.nc文件")
    print("  2. 将脚本中的 USE_REAL_DATA 设置为 True")
    print("  3. 修改 PSL_FILE 为正确的文件路径")