import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from tools import *
from model import factor_selection_lgbm



file_path = "../temp_data/processedData/202508_periods_60.parquet"

print("读取数据...")
df = load_data(file_path)

print(f"数据量: {len(df)}")
print(f"股票数量: {df['code'].nunique()}")
print(f"时间范围: {df['DateTime'].min()} 到 {df['DateTime'].max()}")

# 检查目标变量分布
print(f"\n目标变量统计:")
print(df['future_return'].describe())


# 设置LightGBM的配置参数
config = {
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
}


# 因子筛选
print("\nLightGBM启动！！！")
top_factors, importance_df, model = factor_selection_lgbm(df, config, top_n=30)

# 保存结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = './result'
# 1. 保存top_factors为JSON
top_factors_path = os.path.join(save_dir, f"top_factors_{timestamp}.json")
with open(top_factors_path, 'w', encoding='utf-8') as f:
    json.dump({
        'top_factors': top_factors,
        'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'count': len(top_factors)
    }, f, ensure_ascii=False, indent=2)
# 2. 保存importance_df为CSV
importance_path = os.path.join(save_dir, f"feature_importance_{timestamp}.csv")
importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')