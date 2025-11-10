import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class MinuteFactorCalculator:
    def __init__(self, minute_data, formulas, names, start=None, end=None):
        """
        分钟级数据因子计算
        minute_data:包含open, high, low, close的DataFrame
        formulas:因子表达式列表
        names:因子名称列表
        start:从df中的第几支股票开始处理(第一支股票的start=0)
        end:处理到第几支股票
        """
        self.minute_data = minute_data.copy()
        self.factors_data = None
        self.factor_formulas = formulas
        self.factor_names = names
        self._preprocess_data()
        self.start = start
        self.end = end
        
    
    def _preprocess_data(self):
        """数据预处理"""
        df = self.minute_data
        
        # 确保数据按时间和股票代码排序
        if 'DateTime' in df.columns and 'code' in df.columns:
            df = df.sort_values(['code', 'DateTime']).reset_index(drop=True)
        elif 'DateTime' in df.columns:
            df = df.sort_values('DateTime').reset_index(drop=True)
            
        # 处理价格数据中的异常值
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                # 将0或负值替换为NaN
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].apply(lambda x: x if x > 0 else np.nan)
        
        self.minute_data = df
        
        
    def calculate_factors(self):
        """
        计算指定的因子 - 支持一个df里有多支股票
        factor_formulas:因子公式列表
        """
        factor_formulas = self.factor_formulas.copy()
        factor_names = self.factor_names.copy()
        df = self.minute_data.copy()
        print(f"一共计算 {len(factor_formulas)} 个因子")
        
        
        # 检查是否有股票代码列
        if 'code' not in df.columns:
            print("警告: 数据中没有'code'列, 将按单只股票处理")
            return self._calculate_single_stock_factors(df, factor_formulas, factor_names)
        
        # 按股票分组计算因子
        results = []
        stock_codes = df['code'].unique()
        if self.start == None:
            self.start = 0
        if self.end == None:
            self.end = len(stock_codes)
        print(f"共处理 {self.end - self.start} 支股票")
        for code in tqdm(stock_codes[self.start:self.end]):
            stock_data = df[df['code'] == code].copy()
            
            # 计算单支股票的因子
            stock_factors = self._calculate_single_stock_factors(stock_data, factor_formulas, factor_names)
            results.append(stock_factors)
        
        # 合并所有股票的结果
        self.factors_data = pd.concat(results, ignore_index=True)
        return self.factors_data
        
        
    def _calculate_single_stock_factors(self, df, factor_formulas, factor_names):
        """
        计算单支股票的因子
        """
        # 预先算好，一次性添加到DataFrame(性能优化)
        factor_results = {}
        
        # 计算每个因子
        for i, formula in enumerate(factor_formulas):
            try:
                factor_name = factor_names[i]
                factor_results[factor_name] = self._calculate_single_factor(df, formula)
                # print(f"因子 {factor_name} 计算完毕, \t其表达式为:{formula}")
            except Exception as e:
                print(f"因子 {factor_name} 计算失败,错误: {e}")
                df[factor_names[i]] = np.nan
                
        # 一次性将所有因子列添加到DataFrame
        factors_df = pd.DataFrame(factor_results, index=df.index)

        # 使用pd.concat一次性合并所有因子列，避免内存碎片化
        result_df = pd.concat([df, factors_df], axis=1)
        return result_df
    
    def _calculate_single_factor(self, df, formula):
        """
        计算单个因子
        df:分钟级数据表
        formula:因子公式
        """
        # 替换公式中的函数名
        formula = formula.replace('greater', 'np.maximum')
        formula = formula.replace('less', 'np.minimum')
        formula = formula.replace('Abs', 'np.abs')
        formula = formula.replace('log', 'np.log')
        
        # 定义shift函数
        def shift(series, n):
            """实现shift功能"""
            return series.shift(n)
        
        # 定义分组shift函数
        def group_shift(series, n, group_col=None):
            """如果提供了分组列，按组分shift"""
            if group_col is not None and group_col in df.columns:
                return series.groupby(df[group_col]).shift(n)
            else:
                return series.shift(n)
            
        # 定义滚动计算函数
        def mean(series, window):
            """滚动均值"""
            return series.rolling(window=window).mean()
        
        def std(series, window):
            """滚动标准差"""
            return series.rolling(window=window).std()
        
        def max(series, window):
            """滚动最大值"""
            return series.rolling(window=window).max()
        
        def min(series, window):
            """滚动最小值"""
            return series.rolling(window=window).min()
        
        def sum(series, window):
            """滚动求和"""
            return series.rolling(window=window).sum()
        
        def quantile(series, window, q):
            """滚动分位数"""
            return series.rolling(window=window).quantile(q)
        
        def idxmax(series, window):
            """滚动窗口内最大值的索引位置（从0开始）"""
            def _idxmax(x):
                if len(x) == window and not x.isna().any():
                    return np.argmax(x)
                return np.nan
            return series.rolling(window=window).apply(_idxmax, raw=False)
        
        def idxmin(series, window):
            """滚动窗口内最小值的索引位置（从0开始）"""
            def _idxmin(x):
                if len(x) == window and not x.isna().any():
                    return np.argmin(x)
                return np.nan
            return series.rolling(window=window).apply(_idxmin, raw=False)
        
        def corr(series1, series2, window):
            """滚动相关系数"""
            return series1.rolling(window=window).corr(series2)    
        
        # 准备计算环境
        local_vars = {
            'np': np,
            'pd': pd,
            'shift': shift,
            'group_shift': group_shift,
            'mean': mean,
            'std': std,
            'max': max,
            'min': min,
            'sum': sum,
            'quantile': quantile,
            'idxmax': idxmax,
            'idxmin': idxmin,
            'corr': corr,
            'open': df['open'],
            'high': df['high'], 
            'low': df['low'],
            'close': df['close'],
            'volume': df['volume']
        }
        
        # 如果数据中有code列，也加入到计算环境
        if 'code' in df.columns:
            local_vars['code'] = df['code']
            
        # 安全地计算表达式
        try:
            result = eval(formula, {'__builtins__': {}}, local_vars)
            return result
        except Exception as e:
            print(f"计算表达式失败: {formula}, 错误: {e}")
            return np.nan

    def get_factors_by_stock(self, stock_code):
            """
            获取指定股票的因子数据
            """
            if self.factors_data is None:
                print("请先调用 calculate_factors() 方法计算因子")
                return None

            if 'code' not in self.factors_data.columns:
                print("数据中没有股票代码信息")
                return self.factors_data

            return self.factors_data[self.factors_data['code'] == stock_code].copy()
        
    
def Alpha158_factor():
    # 创建因子表达式和因子名
    # 创建因子计算表达式与因子名称
    
    # 第一部分: 基础因子(9个)
    formulas = ["(close-open)/open",
                "(high-low)/open", 
                "(close-open)/(high-low+1e-12)",
                "(high-np.maximum(open, close))/open",
                "(high-np.maximum(open, close))/(high-low+1e-12)",
                "(np.minimum(open, close)-low)/open",
                "(np.minimum(open, close)-low)/(high-low+1e-12)",
                "(2*close-high-low)/open",
                "(2*close-high-low)/(high-low+1e-12)"]
    names = ["KMID",
             "KLEN",
             "KMID2",
             "KUP",
             "KUP2",
             "KLOW",
             "KLOW2",
             "KSFT",
             "KSFT2"]
    
    # 2: 价格因子(4*5=20个)
    feature = ['open', 'high', 'low', 'close']
    windows = range(5)
    for field in feature:
        field = field.lower()
        formulas += ["shift(%s, %d)/close" % (field, d) if d != 0 else "%s/close" % field for d in windows]
        names += [field.upper() + str(d) for d in windows]
    
    
    # 3: 成交量因子(5个)
    formulas += ["shift(volume, %d)/(volume+1e-12)" % d if d != 0 else "volume/(volume+1e-12)" for d in windows]
    names += ["VOLUME" + str(d) for d in windows]
    
    # 4: 滚动因子(5*5=25个)
    windows = [5, 10, 20, 30, 60]
    formulas += ["shift(close, %d)/close" % d for d in windows]
    names += ["ROC%d" % d for d in windows]

    formulas += ["mean(close, %d)/close" % d for d in windows]
    names += ["MA%d" % d for d in windows]

    formulas += ["std(close, %d)/close" % d for d in windows]
    names += ["STD%d" % d for d in windows]

    formulas += ["max(high, %d)/close" % d for d in windows]
    names += ["MAX%d" % d for d in windows]

    formulas += ["min(low, %d)/close" % d for d in windows]
    names += ["MIN%d" % d for d in windows]
    
    # 其他因子
    formulas += ["quantile(close, %d, 0.8)/close" % d for d in windows]
    names += ["QTLU%d" % d for d in windows]

    formulas += ["quantile(close, %d, 0.2)/close" % d for d in windows]
    names += ["QTLD%d" % d for d in windows]

    formulas += ["(close-min(low, %d))/(max(high, %d)-min(low, %d)+1e-12)" % (d, d, d) for d in windows]
    names += ["RSV%d" % d for d in windows]

    formulas += ["idxmax(high, %d)/%d" % (d, d) for d in windows]
    names += ["IMAX%d" % d for d in windows]

    formulas += ["idxmin(low, %d)/%d" % (d, d) for d in windows]
    names += ["IMIN%d" % d for d in windows]

    formulas += ["(idxmax(high, %d)-idxmin(low, %d))/%d" % (d, d, d) for d in windows]
    names += ["IMXD%d" % d for d in windows]

    formulas += ["corr(close, log(volume+1), %d)" % d for d in windows]
    names += ["CORR%d" % d for d in windows]

    formulas += ["corr(close/shift(close,1), log(volume/shift(volume, 1)+1), %d)" % d for d in windows]
    names += ["CORD%d" % d for d in windows]

    formulas += ["mean(close>shift(close, 1), %d)" % d for d in windows]
    names += ["CNTP%d" % d for d in windows]

    formulas += ["mean(close<shift(close, 1), %d)" % d for d in windows]
    names += ["CNTN%d" % d for d in windows]

    formulas += ["mean(close>shift(close, 1), %d)-mean(close<shift(close, 1), %d)" % (d, d) for d in windows]
    names += ["CNTD%d" % d for d in windows]

    formulas += [
        "sum(greater(close-shift(close, 1), 0), %d)/(sum(Abs(close-shift(close, 1)), %d)+1e-12)" % (d, d)
        for d in windows
    ]
    names += ["SUMP%d" % d for d in windows]

    formulas += [
        "sum(greater(shift(close, 1)-close, 0), %d)/(sum(Abs(close-shift(close, 1)), %d)+1e-12)" % (d, d)
        for d in windows
    ]
    names += ["SUMN%d" % d for d in windows]

    formulas += [
        "(sum(greater(close-shift(close, 1), 0), %d)-sum(greater(shift(close, 1)-close, 0), %d))"
        "/(sum(Abs(close-shift(close, 1)), %d)+1e-12)" % (d, d, d)
        for d in windows
    ]
    names += ["SUMD%d" % d for d in windows]

    formulas += ["mean(volume, %d)/(volume+1e-12)" % d for d in windows]
    names += ["VMA%d" % d for d in windows]

    formulas += ["std(volume, %d)/(volume+1e-12)" % d for d in windows]
    names += ["VSTD%d" % d for d in windows]

    formulas += [
        "std(Abs(close/shift(close, 1)-1)*volume, %d)/(mean(Abs(close/shift(close, 1)-1)*volume, %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["WVMA%d" % d for d in windows]

    formulas += [
        "sum(greater(volume-shift(volume, 1), 0), %d)/(sum(Abs(volume-shift(volume, 1)), %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["VSUMP%d" % d for d in windows]

    formulas += [
        "sum(greater(shift(volume, 1)-volume, 0), %d)/(sum(Abs(volume-shift(volume, 1)), %d)+1e-12)"
        % (d, d)
        for d in windows
    ]
    names += ["VSUMN%d" % d for d in windows]

    formulas += [
        "(sum(greater(volume-shift(volume, 1), 0), %d)-sum(greater(shift(volume, 1)-volume, 0), %d))"
        "/(sum(Abs(volume-shift(volume, 1)), %d)+1e-12)" % (d, d, d)
        for d in windows
    ]
    names += ["VSUMD%d" % d for d in windows]
    
    return formulas, names