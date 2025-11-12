import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def factor_selection_lgbm(df, top_n=30):
    """
    使用LightGBM进行因子筛选，返回前top_n个重要因子
    """
    # 按时间排序
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    # 准备特征和目标变量
    feature_columns = [col for col in df.columns if col not in ['code', 'DateTime', 'future_return']]
    X = df[feature_columns]
    y = df['future_return']
    
    # 检查和处理缺失值
    print(f"数据缺失情况:")
    print(f"特征缺失: {X.isnull().sum().sum()}")
    print(f"目标变量缺失: {y.isnull().sum()}")
    
    # 现在先用0填充，后续数据在交给模型前就要处理好一切的NaN
    X = X.fillna(0)
    y = y.fillna(0)  
    
    # 按时间划分训练集和测试集
    split_point = int(len(df) * 0.8)
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    test_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)
    
    # LightGBM参数
    params = config
    
    # 训练模型
    print("训练LightGBM模型中...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # 模型评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n模型性能:")
    print(f"测试集MSE: {mse:.6f}")
    print(f"测试集R²: {r2:.4f}")
    
    # 获取因子重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importance(importance_type='gain')
    })
    
    # 按重要性排序
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # 选择前top_n个因子
    top_factors = feature_importance.head(top_n)['feature'].tolist()
    
    print(f"\n前{top_n}个重要因子:")
    for i, factor in enumerate(top_factors, 1):
        importance_val = feature_importance[feature_importance['feature'] == factor]['importance'].values[0]
        print(f"{i:2d}. {factor}: {importance_val:.4f}")
    
    return top_factors, feature_importance, model