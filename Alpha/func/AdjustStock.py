import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../log/adjustment_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def adjust_stock_data(df1, df2):
    """
    对分钟级股票数据进行复权处理
    
    输入参数:
        df1: 分钟级股票数据
        df2: 每日复权因子数据
    
    返回值:
        adjusted_df: 复权后的数据
    """
    cur_time = datetime.now()
    print(f"{cur_time}:开始处理...")
    # 将DateTime转换为日期格式，用于匹配
    df1['DateTime'] = pd.to_datetime(df1['DateTime'], format='%Y%m%d %H%M%S')
    print(1)
    df1['trade_date'] = df1['DateTime'].dt.strftime('%Y-%m-%d')
    print(2)
    # 确保df2的data列是datetime格式
    df2['date'] = pd.to_datetime(df2['date']).dt.strftime('%Y-%m-%d')
    print(3)
    # 记录未找到复权因子的情况
    missing_factors = set()
    print(4)
    # 获取所有以为的股票代码和日期组合
    unique_combin = df1[['code', 'trade_date']].drop_duplicates()
    print(5)
    # 创建字典存储复权因子，避免重复查询，提高效率
    factor_cache = {}
    print(6)
    # 预先为每个组合查找复权因子
    cur_time = datetime.now()
    print(f"{cur_time}:预查询本月股票的因子数据...")
    for idx, (_, row) in tqdm(enumertate(unique_combin.iterrows()),total=len(unique_combin), desc='查询复权因子'):
        code = row['code']
        date = row['trade_date']
        cache_key = f"{code}_{date}"
        
        # 如果已在缓存中，跳过
        if cache_key in factor_cache:
            continue
        
        # 否则查找复权因子
        factor_data = df2[(df2['code'] == code) & (df2['date'] == date)]
        
        # 记录未找到复权因子的情况
        if len(factor_data) == 0:
            missing_factors.add((code, date))
            factor_cache[cache_key] = None
        else:
            # 使用平均复权因子
            adjust_factor = factor_data['avg_adjust_factor'].iloc[0]
            factor_cache[cache_key] = adjust_factor
    
    cur_time = datetime.now()
    print(f"{cur_time}:预查询完毕...")
    # 记录未找到复权因子的情况
    if missing_factors:
        for code, date in sorted(missing_factors):
            logging.warning(f'股票{code}在日期 {date} 未查找到复权因子')

    cur_time = datetime.now()
    print(f"{cur_time}:对本月数据进行复权...")
    # 应用复权因子的函数
    def apply_adjustment(row):
        cache_key = f"{row['code']}_{row['trade_date']}"
        adjust_factor = factor_cache.get(cache_key)
        
        if adjust_factor is None or adjust_factor == 1.0:
            return row['volume'], row['open'], row['high'], row['low'], row['close']
        
        # 价格复权：价格乘以后复权因子
        # 成交量复权：成交量除以后复权因子
        adjusted_volume = row['volume'] * adjust_factor
        adjusted_open = row['open'] / adjust_factor
        adjusted_high = row['high'] / adjust_factor
        adjusted_low = row['low'] / adjust_factor
        adjusted_close = row['close'] / adjust_factor
        
        return adjusted_volume, adjusted_open, adjusted_high, adjusted_low, adjusted_close
    
    # 应用福泉处理
    logging.info("开始应用复权因子")
    adjustment_results = df1.apply(apply_adjustment, axis=1, result_type='expand')
    adjustment_results.columns = ['volume','open','high','low','close']
    
    df1['open'] = adjustment_results['open']
    df1['close'] = adjustment_results['close']
    df1['high'] = adjustment_results['high']
    df1['low'] = adjustment_results['low']
    df1['volume'] = adjustment_results['volume']
    
    
    logging.info(f"复权处理完成。共处理 {len(df1)} 行数据")
    logging.info(f"未找到复权因子的股票-日期组合数量: {len(missing_factors)}")
    
    return adjustment_results



def adjust_stock_data_fast(df1, df2):
    """
    高效复权处理 - 使用merge向量化操作
    """
    cur_time = datetime.now()
    print(f"{cur_time}:开始处理...")
    
    # 复制数据
    df1_adj = df1.copy()
    
    # 数据预处理 - 提取日期
    df1_adj['DateTime'] = pd.to_datetime(df1_adj['DateTime'], format='%Y%m%d %H%M%S')
    df1_adj['trade_date'] = df1_adj['DateTime'].dt.strftime('%Y-%m-%d')
    
    # 确保df2的日期格式一致
    df2['date'] = pd.to_datetime(df2['date']).dt.strftime('%Y-%m-%d')

    print(f"{datetime.now()}:开始合并数据...")
    
    # 方法1：直接merge，最快速
    df_merged = df1_adj.merge(
        df2[['code', 'date', 'avg_adjust_factor']], 
        left_on=['code', 'trade_date'], 
        right_on=['code', 'date'], 
        how='left'
    )
    
    # 记录未找到复权因子的情况
    missing_mask = df_merged['avg_adjust_factor'].isna()
    if missing_mask.any():
        missing_data = df_merged[missing_mask][['code', 'trade_date']].drop_duplicates()
        for _, row in missing_data.iterrows():
            logging.warning(f'股票 {row["code"]} 在日期 {row["trade_date"]} 未查找到复权因子')
        print(f"未找到复权因子的记录数: {missing_mask.sum()}")
    
    print(f"{datetime.now()}:开始应用复权...")
    
    # 向量化复权操作
    # 复制需要复权的列
    adjust_mask = (~missing_mask) & (df_merged['avg_adjust_factor'] != 1.0)
    
    # 批量处理 - 使用numpy数组提高速度
    if adjust_mask.any():
        factor_values = df_merged.loc[adjust_mask, 'avg_adjust_factor'].values
        volume_values = df_merged.loc[adjust_mask, 'volume'].values
        open_values = df_merged.loc[adjust_mask, 'open'].values
        high_values = df_merged.loc[adjust_mask, 'high'].values
        low_values = df_merged.loc[adjust_mask, 'low'].values
        close_values = df_merged.loc[adjust_mask, 'close'].values
        
        # 应用复权
        df_merged.loc[adjust_mask, 'volume'] = volume_values * factor_values
        df_merged.loc[adjust_mask, 'open'] = open_values / factor_values
        df_merged.loc[adjust_mask, 'high'] = high_values / factor_values
        df_merged.loc[adjust_mask, 'low'] = low_values / factor_values
        df_merged.loc[adjust_mask, 'close'] = close_values / factor_values
    
    # 清理临时列
    df_merged = df_merged.drop(['trade_date', 'date', 'avg_adjust_factor'], axis=1)
    
    print(f"{datetime.now()}:复权处理完成! 共处理 {len(df_merged)} 行数据")
    
    return df_merged

def adjust_stock_data_optimal(df1, df2):
    """
    最优化的复权处理 - 使用groupby + transform
    """
    cur_time = datetime.now()
    print(f"{cur_time}:开始最优处理...")
    
    # 复制数据
    df1_adj = df1.copy()
    
    # 数据预处理
    df1_adj['DateTime'] = pd.to_datetime(df1_adj['DateTime'], format='%Y%m%d %H%M%S')
    df1_adj['trade_date'] = df1_adj['DateTime'].dt.strftime('%Y-%m-%d')
    df2['date'] = pd.to_datetime(df2['date']).dt.strftime('%Y-%m-%d')
    
    print(f"{datetime.now()}:创建复权因子映射...")
    
    # 创建(code, date)到因子的映射字典 - 一次性创建
    factor_dict = df2.set_index(['code', 'date'])['avg_adjust_factor'].to_dict()
    
    # 为df1创建复合索引用于映射
    df1_adj['factor_key'] = list(zip(df1_adj['code'], df1_adj['trade_date']))
    df1_adj['adjust_factor'] = df1_adj['factor_key'].map(factor_dict)
    
    # 记录缺失因子
    missing_mask = df1_adj['adjust_factor'].isna()
    if missing_mask.any():
        missing_info = df1_adj[missing_mask][['code', 'trade_date']].drop_duplicates()
        for _, row in missing_info.iterrows():
            logging.warning(f'股票 {row["code"]} 在日期 {row["trade_date"]} 未查找到复权因子')
        print(f"缺失因子的唯一组合数: {len(missing_info)}")
    
    print(f"{datetime.now()}:批量应用复权...")
    
    # 批量复权操作 - 使用numpy向量化
    adjust_mask = (~missing_mask) & (df1_adj['adjust_factor'] != 1.0)
    
    if adjust_mask.any():
        # 一次性处理所有需要复权的数据
        factors = df1_adj.loc[adjust_mask, 'adjust_factor'].values
        volume = df1_adj.loc[adjust_mask, 'volume'].values
        open_vals = df1_adj.loc[adjust_mask, 'open'].values
        high_vals = df1_adj.loc[adjust_mask, 'high'].values
        low_vals = df1_adj.loc[adjust_mask, 'low'].values
        close_vals = df1_adj.loc[adjust_mask, 'close'].values
        
        # 应用复权
        df1_adj.loc[adjust_mask, 'volume'] = volume * factors
        df1_adj.loc[adjust_mask, 'open'] = open_vals / factors
        df1_adj.loc[adjust_mask, 'high'] = high_vals / factors
        df1_adj.loc[adjust_mask, 'low'] = low_vals / factors
        df1_adj.loc[adjust_mask, 'close'] = close_vals / factors
    
    # 清理临时列
    df1_adj = df1_adj.drop(['trade_date', 'factor_key', 'adjust_factor'], axis=1)
    
    print(f"{datetime.now()}:处理完成! 共 {len(df1_adj)} 行数据")
    
    return df1_adj