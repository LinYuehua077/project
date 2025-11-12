import pandas as pd
import os
from datetime import datetime, timedelta


def findCodeColname(df, code_col=None):
    """
    寻找股票代码列名(兼容各种常见的股票代码命名形式)
        参数:
            df: 包含股票代码列的股票数据DataFrame
            date_col: 手动指定的股票代码列名（可选，为None或列名不存在时自动匹配）
        返回:
            str: 匹配到的股票代码列名
        异常:
            ValueError: 未找到任何常见股票代码列时抛出
    """
    if code_col is None or code_col not in df.columns:
        # 常见的股票代码列名，可持续更新
        common_code_cols = [
            'code', 'stock_code', '股票代码', '证券代码', 'sec_code', 
            'ticker', 'ticker_symbol', '个股代码', 'code_stock'
        ]
        # 筛选df中存在的列名
        matched_cols = [col for col in common_code_cols if col in df.columns]
        if not matched_cols:
            raise ValueError(
                f"未找到股票代码列！请手动指定code_col参数。\n"
                f"df中所有列名：{df.columns.tolist()}\n"
                f"函数支持的常见列名：{common_code_cols}"
            )
        # 优先选择第一个匹配的列名
        code_col = matched_cols[0]
        if len(matched_cols)>1:
            print(f"匹配到多个股票代码列，请检查：{matched_cols}")
    return code_col

def findDateColname(df, date_col=None):
    """
    寻找股票数据中的时间列名（兼容各种常见的时间列命名形式）
        参数:
            df: 包含时间列的股票数据DataFrame
            date_col: 手动指定的时间列名（可选，为None或列名不存在时自动匹配）
        返回:
            str: 匹配到的时间列名
        异常:
            ValueError: 未找到任何常见时间列时抛出
    """
    # 若手动指定的列名存在，直接返回
    if date_col is not None and date_col in df.columns:
        return date_col
    
    # 常见的股票数据时间列名（覆盖A股、港股、美股常用命名，可持续更新）
    common_date_cols = [
        'date', '日期', '交易日期', 'trade_date', '成交日期',
        'datetime', 'time', 'trading_date', 'transaction_date', 'DateTime',
        'trade_dt', 'dt', 't_date', 'trans_date',
        'trade_date_', 'hk_trade_date', 'us_trade_date'
    ]
    
    # 筛选df中实际存在的时间列名
    matched_cols = [col for col in common_date_cols if col in df.columns]
    
    # 处理匹配结果
    if not matched_cols:
        raise ValueError(
            f"未找到时间列！请手动指定date_col参数。\n"
            f"df中所有列名：{df.columns.tolist()}\n"
            f"函数支持的常见时间列名：{common_date_cols}"
        )
    # 匹配到多个时给出提示，避免误选
    elif len(matched_cols) > 1:
        print(f"匹配到多个可能的时间列，请确认实际列名：{matched_cols}")
    
    # 优先返回第一个匹配的列名（按common_date_cols优先级排序）
    return matched_cols[0]
    

def getTradingDaysInMonth(file_path_lst, file_type, year, month):
    """
    获取指定月份中所有交易日期
    """
    trading_days = []
    for file_path in file_path_lst:
        # 从路径中提取文件名，也就是提取文件日期
        filename = os.path.basename(file_path)
        if filename.endswith(f'{file_type}'):
            try:
                file_date_str = filename.replace(f".{file_type}", "")
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
                if file_date.year == year and file_date.month == month:
                    trading_days.append(file_date.date())
            except ValueError:
                print(f"文件 {file_path} 命名不合规")
                continue
    return sorted(trading_days)

def getLastTradingDayOfPreviousMonth(file_path_lst, year, month, file_type='parquet'):
    """
    从文件路径列表中获取指定年月的前一个月的最后一个交易日
    
    输入参数:
        - file_path_lst: 包含所有数据文件绝对路径的列表
        - year: 目标年份
        - month: 目标月份
    返回值
        如果有前一个月的数据则返回该日期，否则返回None(第一个月肯定是None)
    """
    if month == 1:
        prev_year = year - 1
        prev_month = 12
    else:
        prev_year = year
        prev_month = month - 1
    
    prev_month_days = getTradingDaysInMonth(file_path_lst, file_type, prev_year, prev_month)
    return max(prev_month_days) if prev_month_days else None


def getFirstTradingDayOfNextMonth(file_path_lst, year, month, file_type='parquet'):
    """
    从文件路径列表中获取指定年月的下一个月的第一个交易日
    
    输入参数:
        - file_path_lst: 包含所有数据文件绝对路径的列表
        - year: 目标年份
        - month: 目标月份
    返回值
        如果有下一个月的数据则返回该日期，否则返回None(最后一个月肯定是None)
    """
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1
    
    next_month_days = getTradingDaysInMonth(file_path_lst, file_type, next_year, next_month)
    return min(next_month_days) if next_month_days else None



def loadDailyData(file_path, data_type):
    """
    加载指定文件路径的分钟级交易数据
    """
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        # date_colname = findDateColname(df)
        # df[date_colname] = pd.to_datetime(df[date_colname], format='%Y%m%d %H%M%S')
        if data_type.lower() == 'data':
            return df[['code', 'DateTime', 'volume', 'turnover']]
        if data_type.lower() == 'price':
            return df[['code', 'DateTime', 'open', 'high', 'low', 'close']]    
    print(f"警告1: 文件 {file_path} 不存在")
    return None
    
    
def getFilePathByDate(file_path_lst, target_date, file_type='parquet'):
    """
    根据日期从文件路径列表中查找对应的文件路径
    
    Parameters:
    - file_path_lst: 包含所有数据文件绝对路径的列表
    - target_date: 目标日期 (datetime.date对象)
    """
    date_str = target_date.strftime('%Y%m%d')
    
    for file_path in file_path_lst:
        filename = os.path.basename(file_path)
        if filename.startswith(date_str) and filename.endswith(file_type):
            return file_path
    
    return None


import pandas as pd
import os
from datetime import datetime, timedelta


def findCodeColname(df, code_col=None):
    """
    寻找股票代码列名(兼容各种常见的股票代码命名形式)
        参数:
            df: 包含股票代码列的股票数据DataFrame
            date_col: 手动指定的股票代码列名（可选，为None或列名不存在时自动匹配）
        返回:
            str: 匹配到的股票代码列名
        异常:
            ValueError: 未找到任何常见股票代码列时抛出
    """
    if code_col is None or code_col not in df.columns:
        # 常见的股票代码列名，可持续更新
        common_code_cols = [
            'code', 'stock_code', '股票代码', '证券代码', 'sec_code', 
            'ticker', 'ticker_symbol', '个股代码', 'code_stock'
        ]
        # 筛选df中存在的列名
        matched_cols = [col for col in common_code_cols if col in df.columns]
        if not matched_cols:
            raise ValueError(
                f"未找到股票代码列！请手动指定code_col参数。\n"
                f"df中所有列名：{df.columns.tolist()}\n"
                f"函数支持的常见列名：{common_code_cols}"
            )
        # 优先选择第一个匹配的列名
        code_col = matched_cols[0]
        if len(matched_cols)>1:
            print(f"匹配到多个股票代码列，请检查：{matched_cols}")
    return code_col

def findDateColname(df, date_col=None):
    """
    寻找股票数据中的时间列名（兼容各种常见的时间列命名形式）
        参数:
            df: 包含时间列的股票数据DataFrame
            date_col: 手动指定的时间列名（可选，为None或列名不存在时自动匹配）
        返回:
            str: 匹配到的时间列名
        异常:
            ValueError: 未找到任何常见时间列时抛出
    """
    # 若手动指定的列名存在，直接返回
    if date_col is not None and date_col in df.columns:
        return date_col
    
    # 常见的股票数据时间列名（覆盖A股、港股、美股常用命名，可持续更新）
    common_date_cols = [
        'date', '日期', '交易日期', 'trade_date', '成交日期',
        'datetime', 'time', 'trading_date', 'transaction_date', 'DateTime',
        'trade_dt', 'dt', 't_date', 'trans_date',
        'trade_date_', 'hk_trade_date', 'us_trade_date'
    ]
    
    # 筛选df中实际存在的时间列名
    matched_cols = [col for col in common_date_cols if col in df.columns]
    
    # 处理匹配结果
    if not matched_cols:
        raise ValueError(
            f"未找到时间列！请手动指定date_col参数。\n"
            f"df中所有列名：{df.columns.tolist()}\n"
            f"函数支持的常见时间列名：{common_date_cols}"
        )
    # 匹配到多个时给出提示，避免误选
    elif len(matched_cols) > 1:
        print(f"匹配到多个可能的时间列，请确认实际列名：{matched_cols}")
    
    # 优先返回第一个匹配的列名（按common_date_cols优先级排序）
    return matched_cols[0]
    

def getTradingDaysInMonth(file_path_lst, file_type, year, month):
    """
    获取指定月份中所有交易日期
    """
    trading_days = []
    for file_path in file_path_lst:
        # 从路径中提取文件名，也就是提取文件日期
        filename = os.path.basename(file_path)
        if filename.endswith(f'{file_type}'):
            try:
                file_date_str = filename.replace(f".{file_type}", "")
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
                if file_date.year == year and file_date.month == month:
                    trading_days.append(file_date.date())
            except ValueError:
                print(f"文件 {file_path} 命名不合规")
                continue
    return sorted(trading_days)

def getLastTradingDayOfPreviousMonth(file_path_lst, year, month, file_type='parquet'):
    """
    从文件路径列表中获取指定年月的前一个月的最后一个交易日
    
    输入参数:
        - file_path_lst: 包含所有数据文件绝对路径的列表
        - year: 目标年份
        - month: 目标月份
    返回值
        如果有前一个月的数据则返回该日期，否则返回None(第一个月肯定是None)
    """
    if month == 1:
        prev_year = year - 1
        prev_month = 12
    else:
        prev_year = year
        prev_month = month - 1
    
    prev_month_days = getTradingDaysInMonth(file_path_lst, file_type, prev_year, prev_month)
    return max(prev_month_days) if prev_month_days else None

def getFirstTradingDayOfNextMonth(file_path_lst, year, month, file_type='parquet'):
    """
    从文件路径列表中获取指定年月的下一个月的第一个交易日
    
    输入参数:
        - file_path_lst: 包含所有数据文件绝对路径的列表
        - year: 目标年份
        - month: 目标月份
    返回值
        如果有下一个月的数据则返回该日期，否则返回None(最后一个月肯定是None)
    """
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1
    
    next_month_days = getTradingDaysInMonth(file_path_lst, file_type, next_year, next_month)
    return min(next_month_days) if next_month_days else None


def loadDailyData(file_path, data_type):
    """
    加载指定文件路径的分钟级交易数据
    """
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        # date_colname = findDateColname(df)
        # df[date_colname] = pd.to_datetime(df[date_colname], format='%Y%m%d %H%M%S')
        if data_type.lower() == 'data':
            return df[['code', 'DateTime', 'volume', 'turnover']]
        if data_type.lower() == 'price':
            return df[['code', 'DateTime', 'open', 'high', 'low', 'close']]    
    print(f"警告1: 文件 {file_path} 不存在")
    return None
    
    
def getFilePathByDate(file_path_lst, target_date, file_type='parquet'):
    """
    根据日期从文件路径列表中查找对应的文件路径
    
    Parameters:
    - file_path_lst: 包含所有数据文件绝对路径的列表
    - target_date: 目标日期 (datetime.date对象)
    """
    date_str = target_date.strftime('%Y%m%d')
    
    for file_path in file_path_lst:
        filename = os.path.basename(file_path)
        if filename.startswith(date_str) and filename.endswith(file_type):
            return file_path
    
    return None


def processMonthlyData(file_path_lst, year, month, data_type, file_type='parquet',
                      include_previous_last_day=True, include_next_first_day=True):
    """
    处理指定月份的数据，可选择是否包含上个月最后交易日和下个月第一个交易日的数据
    
    Parameters:
        - file_path_lst: 包含所有数据文件绝对路径的列表
        - year: 目标年份
        - month: 目标月份
        - include_previous_last_day: 是否包含上个月最后交易日的数据
        - include_next_first_day: 是否包含下个月第一个交易日的数据
    """
    monthly_data = []
    
    # 如果需要包含上个月最后一天的数据
    if include_previous_last_day:
        last_day_prev_month = getLastTradingDayOfPreviousMonth(file_path_lst, year, month)
        if last_day_prev_month:
            print(f"加载上个月最后交易日数据: {last_day_prev_month}")
            file_path = getFilePathByDate(file_path_lst, last_day_prev_month)
            if file_path:
                prev_day_data = loadDailyData(file_path, data_type)
                if prev_day_data is not None:
                    monthly_data.append(prev_day_data)
            else:
                print(f"警告2: 未找到日期 {last_day_prev_month} 对应的数据文件")
    
    # 加载当前月份的所有交易日数据
    current_month_days = getTradingDaysInMonth(file_path_lst, file_type, year, month)
    print(f"加载 {year}年{month}月的数据")
    
    for trading_day in current_month_days:
        file_path = getFilePathByDate(file_path_lst, trading_day)
        if file_path:
            day_data = loadDailyData(file_path, data_type)
            if day_data is not None:
                monthly_data.append(day_data)
        else:
            print(f"警告3: 未找到日期 {trading_day} 对应的数据文件")
    
    # 如果需要包含下个月第一天数据
    if include_next_first_day:
        first_day_next_month = getFirstTradingDayOfNextMonth(file_path_lst, year, month)
        if first_day_next_month:
            print(f"加载下个月第一个交易日数据: {first_day_next_month}")
            file_path = getFilePathByDate(file_path_lst, first_day_next_month)
            if file_path:
                next_day_data = loadDailyData(file_path, data_type)
                if next_day_data is not None:
                    monthly_data.append(next_day_data)
            else:
                print(f"警告4: 未找到日期 {first_day_next_month} 对应的数据文件")
    
    if monthly_data:
        # 合并所有数据并按先股票后时间的顺序排序
        code_col = findCodeColname(monthly_data[0])
        date_col = findDateColname(monthly_data[0])
        
        combined_data = pd.concat(monthly_data, ignore_index=True)
        combined_data = combined_data.sort_values([code_col,date_col]).reset_index(drop=True)
        
        print(f"{year}年{month}月数据处理完成，总数据量: {len(combined_data)} 行")
        print(f"时间范围: {combined_data['DateTime'].min()} 到 {combined_data['DateTime'].max()}")
        
        return combined_data
    else:
        print(f"警告5: {year}年{month}月没有找到数据")
        return None

    
def generateMonthRange(start_year, start_month, end_year, end_month):
    """生成从起始年月到结束年月的所有月份列表"""
    months = []
    current_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)

    while current_date <= end_date:
        months.append((current_date.year, current_date.month))
        # 移动到下个月
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return months
    
    
# 使用示例
if __name__ == "__main__":
    # 假设已经有一个包含所有文件路径的列表
    # file_path_lst = ['/path/to/20250504.parquet', '/path/to/20250505.parquet', ...]
    
    file_path_lst = [...]  # 文件路径列表
    
    # 生成从2023年1月到2025年11月的所有月份
    target_months = generateMonthRange(2023, 1, 2025, 11)
    print(f"处理月份范围: {len(target_months)} 个月")
    print(f"从 {target_months[0]} 到 {target_months[-1]}")
    
    
    for i, (year, month) in enumerate(target_months):
        print(f"\n处理第 {i+1}/{len(target_months)} 个周期: {year}年{month}月")
        
        # 判断是否需要包含上个月数据：不是第一个处理周期就需要
        include_previous = (i > 0)
        # 判断是否需要包含下个月数据：不是最后一个处理周期就需要
        include_next = (i < len(target_months) - 1)
        
        monthly_data = processMonthlyData(
            file_path_lst=file_path_lst,
            year=year,
            month=month,
            data_type='data',  # 根据实际情况指定数据类型
            include_previous_last_day=include_previous,
            include_next_first_day=include_next
        )
        
        if monthly_data is not None:
            # 保存结果，文件名包含年份信息
            output_file = f"factor_data_{year}_{month:02d}.csv"
            monthly_data.to_csv(output_file, index=False)
            print(f"数据已保存: {output_file}")
            
            # 清理内存
            del monthly_data

    
def generateMonthRange(start_year, start_month, end_year, end_month):
    """生成从起始年月到结束年月的所有月份列表"""
    months = []
    current_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)

    while current_date <= end_date:
        months.append((current_date.year, current_date.month))
        # 移动到下个月
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return months
    
    
# 使用示例
if __name__ == "__main__":
    # 假设已经有一个包含所有文件路径的列表
    # file_path_lst = ['/path/to/20250504.parquet', '/path/to/20250505.parquet', ...]
    
    file_path_lst = [...]  # 文件路径列表
    
    # 生成从2023年1月到2025年11月的所有月份
    target_months = generate_month_range(2023, 1, 2025, 11)
    print(f"处理月份范围: {len(target_months)} 个月")
    print(f"从 {target_months[0]} 到 {target_months[-1]}")
    
    
    for i, (year, month) in enumerate(target_months):
        print(f"\n处理第 {i+1}/{len(target_months)} 个周期: {year}年{month}月")
        
        # 判断是否需要包含上个月数据：不是第一个处理周期就需要
        include_previous = (i > 0)
        
        monthly_data = process_monthly_data(
            file_path_lst=file_path_lst,
            year=year,
            month=month,
            include_previous_last_day=include_previous
        )
        
        if monthly_data is not None:
            # 保存结果，文件名包含年份信息
            output_file = f"factor_data_{year}_{month:02d}.csv"
            monthly_data.to_csv(output_file, index=False)
            print(f"数据已保存: {output_file}")
            
            # 清理内存
            del monthly_data