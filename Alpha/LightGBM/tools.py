import pandas as pd
import numpy as np
# 读取数据
def load_data(file_path):
    """
    读取股票数据
    """
    df = pd.read_parquet(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(['code', 'DateTime']).reset_index(drop=True)
    return df


# 计算未来n分钟收益率
def calculate_future_return(df, periods=60):
    """
    计算未来n分钟的收益率
    """
    df = df.copy()
    df['future_return'] = df.groupby('code')['close'].shift(-periods) / df['close'] - 1
    return df


# 数据预处理
def preprocess_data(df):
    """
    数据预处理：处理缺失值、异常值等
    """
    
    # 统计 future_return 列的 NaN 数量
    nan_count = df['future_return'].isna().sum()
    if nan_count > 0:
        print(f"警报：future_return列存在{nan_count}个空值")
        # 删除未来收益率为空的数据
        df = df.dropna(subset=['future_return'])
    
    
    # 定义因子列（排除非因子列）
    factor_columns = [col for col in df.columns if col not in ['code', 'DateTime', 'future_return']]

    # 统计每个因子列的缺失值数量
    nan_counts = df[factor_columns].isna().sum()
    # 筛选出有缺失值的列（仅显示非零项）
    nan_cols = nan_counts[nan_counts > 0]
    total_nan = nan_counts.sum()  # 所有因子列的总缺失值数
    # 存在缺失值则触发警报
    if total_nan > 0:
        print(f"警报：因子列中检测到缺失值！")
        print("各因子列缺失值统计：")
        for col, count in nan_cols.items():
            print(f"  - {col}: {count} 个")
        print(f"缺失值总计：{total_nan} 个")
      
        # 处理因子中的缺失值
        df[factor_columns] = df[factor_columns].fillna(method='ffill').fillna(0)
    
    
    # 处理极端异常值(异常值被“修正”到合理范围)
    for col in factor_columns:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = np.clip(df[col], q1, q99)
    
    return df


# 计算IC值（信息系数）
def calculate_ic(df, factors):
    """
    计算每个因子的IC值（皮尔逊相关系数）
    """
    ic_results = []
    
    for factor in factors:
        # 计算每个因子与未来收益率的相关系数
        ic_value = df[factor].corr(df['future_return'])
        ic_results.append({
            'factor': factor,
            'IC': ic_value,
            'abs_IC': abs(ic_value)
        })
    
    ic_df = pd.DataFrame(ic_results)
    ic_df = ic_df.sort_values('abs_IC', ascending=False)
    
    return ic_df

# 计算滚动IC
def calculate_rolling_ic(df, factors, window=252):
    """
    计算滚动IC值
    """
    rolling_ic = pd.DataFrame(index=df['DateTime'].unique())
    
    for factor in factors:
        # 按时间计算滚动相关系数
        temp_df = df[['DateTime', factor, 'future_return']].set_index('DateTime')
        rolling_corr = temp_df[factor].rolling(window=window).corr(temp_df['future_return'])
        rolling_ic[factor] = rolling_corr
    
    rolling_ic = rolling_ic.dropna()
    return rolling_ic


def plot_results(ic_df, rolling_ic):
    """
    可视化结果
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 因子重要性前10
    top_10_importance = ic_df.head(10)
    axes[0, 0].barh(range(10), top_10_importance['abs_IC'])
    axes[0, 0].set_yticks(range(10))
    axes[0, 0].set_yticklabels(top_10_importance['factor'])
    axes[0, 0].set_title('Top 10 Factors by IC Absolute Value')
    axes[0, 0].set_xlabel('|IC|')
    
    # 2. IC值分布
    axes[0, 1].hist(ic_df['IC'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(ic_df['IC'].mean(), color='red', linestyle='--', label=f'Mean: {ic_df["IC"].mean():.4f}')
    axes[0, 1].set_title('IC Value Distribution')
    axes[0, 1].set_xlabel('IC Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. 滚动IC均值（前5个因子）
    top_5_factors = ic_df.head(5)['factor'].tolist()
    rolling_ic_mean = rolling_ic[top_5_factors].mean(axis=1)
    
    axes[1, 0].plot(rolling_ic_mean.index, rolling_ic_mean.values)
    axes[1, 0].set_title('Rolling IC Mean (Top 5 Factors)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Rolling IC')
    
    # 4. IC值热力图
    ic_pivot = ic_df.set_index('factor')['IC'].head(10)
    colors = ['red' if x < 0 else 'blue' for x in ic_pivot.values]
    axes[1, 1].barh(range(len(ic_pivot)), ic_pivot.values, color=colors)
    axes[1, 1].set_yticks(range(len(ic_pivot)))
    axes[1, 1].set_yticklabels(ic_pivot.index)
    axes[1, 1].set_title('Top 10 Factors IC Values (Red: Negative, Blue: Positive)')
    axes[1, 1].set_xlabel('IC Value')
    
    plt.tight_layout()
    plt.savefig('factor_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
