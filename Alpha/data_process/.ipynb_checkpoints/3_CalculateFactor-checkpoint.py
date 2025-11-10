import pandas as pd

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from func.CalculateFactor import MinuteFactorCalculator, Alpha158_factor


# 设置数据集路径
Workspace = "/home/vscode/workspace/data/store/external_history/MintueData/TickData/"
date = 20251020

# 读取表格
df_price = pd.read_parquet(Workspace + f'Price/{date}.parquet')
df_data = pd.read_parquet(Workspace + f'Data/{date}.parquet')

# 聚焦于需要的数据
df_price = df_price[['code', 'DateTime', 'open', 'high', 'low', 'close']].copy()
df_data = df_data[['code', 'DateTime', 'volume', 'turnover']].copy()
df_price_indexed = df_price.set_index(['code', 'DateTime'])
df_data_indexed = df_data.set_index(['code', 'DateTime'])
merged_df = df_price_indexed.join(df_data_indexed, how='left').reset_index()

# 创建计算对象
formulas, names = Alpha158_factor()

cal = MinuteFactorCalculator(merged_df, formulas, names, 0, 20)
cal.calculate_factors()

# 保存计算结果
cal.factors_data.to_excel('./temp_data/factors.xlsx', index=False)

# 查看特定股票的因子数据
stock_factors = cal.get_factors_by_stock('000001.sz')
print(stock_factors)