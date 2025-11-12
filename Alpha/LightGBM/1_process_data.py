import os
import pandas as pd

from tools import *


# 存储因子数据的文件位置
factor_file_lst = os.listdir('../temp_data/factorData')
periods = 60
df_all = pd.DataFrame()
df_lst = []
for factor_file in factor_file_lst:
    target_day = factor_file.split('.')[0]
    target_file_type = factor_file.split('.')[1]
    
    # 因子文件
    factor_file_path = os.path.join('../temp_data/factorData', factor_file)
    df = load_data(factor_file_path)
    
    # 去掉不是本月份的数据
    target_year = int(target_day) // 100  # 2025
    target_month = int(target_day) % 100   # 3
    # 确保df的DataTime格式正确
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    # 筛选：仅保留「目标年+目标月」的行
    df = df[(df['DateTime'].dt.year == target_year) & 
            (df['DateTime'].dt.month == target_month)]
    df_lst.append(df)

print(len(df_lst))
for df in df_lst:
    df_all = pd.concat(df)

df = df_all.sort_values(by=['code', 'DateTime'])
# 计算未来60分钟的收益率
df = calculate_future_return(df)
print(df)


# 数据预处理
df = preprocess_data(df)
processed_file_path = f'../temp_data/processedData/periods_60.parquet'