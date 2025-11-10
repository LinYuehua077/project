import pandas as pd
from tqdm import tqdm

DATA_PATH = "/home/vscode/workspace/data/store/rsync"
path=f'{DATA_PATH}/tonglian_data_daily/tonglian_data_daily.pickle'
path_=f'{DATA_PATH}/tonglian_data_daily/tonglian_stock_day_n.parquet'

import warnings
import pickle
warnings.filterwarnings("ignore")
import os

if 'adjustment_temp.parquet' not in os.listdir('../temp_data/'):
    # 如果 中间表格(temp) 不存在，那么就将数据处理成中间表格
    # 跳过if的内容，即跳过处理中间表格这段代码，进入到后续的分析代码
    # 获取复权数据
    with open(path, 'rb') as f:
        data_daily = pickle.load(f)
    closew=data_daily['closew']
    openw=data_daily['openw']
    highw=data_daily['highw']
    loww=data_daily['loww']
    amtw=data_daily['amtw']

    open = openw[openw.index>="2017-01-01"]
    high = highw[highw.index>="2017-01-01"]
    low = loww[loww.index>="2017-01-01"]
    close = closew[closew.index>="2017-01-01"]
    amount = amtw[amtw.index>="2017-01-01"]

    # 获取未复权数据
    df = pd.read_parquet(path_)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df[df['date'] >= '2017-01-01']
    df = df.reset_index(drop=True)


    def merge_tables_method(table1, table2, new_column='new_column'):
        """
        使用apply函数将复权的openw和closew合并到未复权的df中,从而便于计算复权因子
        """
        # 创建表2的副本
        table2_merged = table2.copy()

        tqdm.pandas(desc=f'处理{new_column}')
        # 添加新列
        table2_merged[new_column] = table2_merged.progress_apply(
            lambda row: table1.loc[row['date'], row['code']] 
            if (row['date'] in table1.index and row['code'] in table1.columns) 
            else None, 
            axis=1
        )

        return table2_merged

    # 将复权开盘价加入df
    df = merge_tables_method(openw, df, 'openw')
    # 将复权收盘价加入df
    df = merge_tables_method(closew, df, 'closew')

    # 将处理好的数据保存一下(因为合并df的处理时间较长)
    df.to_parquet('../temp_data/adjustment_temp.parquet')


# 现在已经有中间表格了
df = pd.read_parquet('../temp_data/adjustment_temp.parquet')

# 使用开盘价计算复权因子
df['adjust_factor_open'] = df['open'] / df['openw']


# 使用收盘价计算复权因子
df['adjust_factor_close'] = df['close'] / df['closew']

# 平均复权因子
df['avg_adjust_factor'] = (df['adjust_factor_open'] + df['adjust_factor_close']) / 2

# 保存计算好的复权因子
df = df[['code','date','adjust_factor_open','adjust_factor_close','avg_adjust_factor']]
df.to_parquet('../temp_data/adjustment.parquet')