import pandas as pd
import os
from tqdm import tqdm

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from func import AdjustStock as adj


# 设置数据集路径
DATA_space = "/home/vscode/workspace/work/ssl/Alpha/temp_data"
merged_path = f"{DATA_space}/mergedData"

# 构建合并数据集路径
merged_data_path_lst = [os.path.join(merged_path, path) for path in sorted(os.listdir(merged_path))]

# 加载后复权因子表
df_fac = pd.read_parquet(f"{DATA_space}/adjustment.parquet")

for file_path in tqdm(merged_data_path_lst):
    df = pd.read_parquet(file_path)
    df_adj = adj.adjust_stock_data_optimal(df, df_fac)
    df_adj.to_parquet(f"{DATA_space}/adjustedData/{file_path.split('/')[-1]}")