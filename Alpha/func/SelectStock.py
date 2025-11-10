import pandas as pd
import os
from datetime import datetime, timedelta

def standardizeCode(code):
    """
    标准化股票代码（统一转为6位纯数字字符串，兼容多格式）
        输入参数：
            code：股票初始代码
        返回值
            code_str：标准化后的形式
    """
    code_str = str(code).strip()
    # 去除市场后缀（.SZ/.SH等）和浮点数后缀（.0）
    if '.' in code_str:
        code_str = code_str.split('.')[0]
    # 过滤非数字字符，补全6位
    code_str = ''.join(filter(str.isdigit, code_str))
    return code_str.zfill(6) if code_str else ''

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
    

def setTimeRange(file_path, file_type, start, end):
    """
    用于筛选某个时间段内的交易数据
        file_path: 存放所有时间的股票数据的文件夹地址
        file_type: 文件的类型(例如parquet,h5,xls,xlsx),可以传递列表,比如['xls', 'xlsx']
        start: 开始日期, 若开始日期没有交易记录, 则往后(未来)自动延申
        end:   结束日期, 若结束日期没有交易记录, 则往前(过去)自动延伸
    返回值:(列表) 按时间升序排列的选中交易记录文件路径列表
    """
    # 日期格式验证与转换(统一成datatime对象,便于筛选)
    date_format = "%Y%m%d"
    start = str(start)
    end = str(end)
    try:
        start_date = datetime.strptime(start, date_format)
        end_date = datetime.strptime(end, date_format)
    except ValueError:
        raise ValueError(f"日期格式错误!请使用'{date_format}'格式,例如(20180518)")
    
    # 校验开始日期不能晚于结束日期
    if start_date > end_date:
        raise ValueError("开始日期不能晚于结束日期")
    
    
    # 获取文件路径下所有文件的日期
    valid_files = {}  # key: datetime对象  value:文件路径
    for filename in os.listdir(file_path):
        # 1. 过滤非指定类型的文件
        # 如果file_type是字符串,则直接判断
        if isinstance(file_type, str):
            # 跳过文件类型不匹配的文件
            if not filename.endswith(f".{file_type}"):
                continue
        # 如果file_type是列表,则
        elif isinstance(file_type, list):
            # 跳过文件类型不匹配的文件
            if not any(filename.endswith(f".{ft}") for ft in file_types):
                continue
        
        # 2. 判断文件前缀是否为合法的日期
        date_str = filename[:-8]  # 截取文件前8位日期字符串
        try:
            file_date = datetime.strptime(date_str, date_format)
            valid_files[file_date] = os.path.join(file_path, filename)
        except (ValueError, IndexError):
            # 跳过命名不规范的文件(如日期位数不对、格式错误等情况)
            continue
        
        # 3. 处理没有获取到任何有效交易记录的特殊情况
        if not valid_files:
            raise FileNotFoundError(f"在路径{file_path}中未找到有效文件,请检查路径或文件命名情况")
        
        
        # 4. 调整开始日期: 无数据则往后顺延到最近的有数据日期
        sorted_dates = sorted(valid_files.keys())  # 按时间升序排列有效日期
        adjusted_start = start_date
        if start_date not in valid_files:
            # 筛选所有大于等于原始开始日期的有效日期
            candidate_starts = [d for d in sorted_dates if d >= start_date]
            if not candidate_starts:
                # 极端情况: 所有数据都早于开始日期,则取最新的一个数据
                adjusted_start = sorted_dates[-1]
            else:
                adjusted_start = candidate_starts[0]
        
        # 5. 调整结束日期: 无数据则往前回溯到最近的有效日期
        adjusted_end = end_date
        if end_date not in valid_files:
            candidate_ends = [d for d in sorted_dates if d <= end_date]
            if not candidate_ends:
                adjusted_end = sorted_dates[0]
            else:
                adjusted_end = candidate_ends[-1]
            
        # 6. 筛选出符合时间范围的所有文件(按时间升序)
        selected_paths = [
            valid_files[date]
            for date in sorted_dates
            if adjusted_start <= date <= adjusted_end
        ]
        
    return selected_paths
        
    
def areFileListsIdentical(lst1, lst2):
    """
    lst1和lst2均为存放文件地址的列表,本函数的功能是判断二者存放的文件名是否一致,前面的路径可以不一致
    """
    adjusted_lst1 = [os.path.basename(f) for f in lst1]
    adjusted_lst2 = [os.path.basename(f) for f in lst2]
    
    return set(adjusted_lst1) == set(adjusted_lst2) and len(lst1) == len(lst2)


def selectByCode(df, target_codes, code_col=None, keep_type=True):
    """
    根据目标股票代码列表筛选DataFrame，兼容多种股票代码格式，返回匹配结果和未找到的代码
    输入参数:
        df: 原始股票数据DataFrame，需包含股票代码列
        target_codes: 目标股票代码列表，支持格式：
              - 带市场后缀：['000001.SZ', '000002.SH']
              - 纯数字字符串：['000001', '000002']、['1', '2']
              - 整数：[1, 2, 000001]（注意：整数1会被转为'000001'补全6位）
        code_col: df中股票代码列的列名
        keep_type: 是否保持原始df中的股票代码风格(即未去掉.SH .SZ .BJ后缀的风格)
    返回值:
        tuple: (匹配目标代码的DataFrame, 未找到的股票代码列表)
    """
    # 找到列名
    code_col = findCodeCol(df, code_col)       
    
    # 预处理目标代码——统一转为「6位纯数字字符串」
    processed_targets = []
    for code in target_codes:
        code_str = standardizeCode(code)
        # 去重（避免目标列表中重复代码）
        if code_str not in processed_targets:
            processed_targets.append(code_str)
            
    # 预处理df中的股票代码列——统一转为「6位纯数字字符串」
    df_processed = df.copy()
    
    # # 处理df中代码：转为字符串→去后缀→补6位→去重
    # df_processed[code_col] = df_processed[code_col].astype(str).apply(
    #     lambda x: x.split('.')[0].strip().zfill(6) if '.' in x else x.strip().zfill(6)
    # )
    
    df_processed[code_col] = df_processed[code_col].astype(str).apply(lambda x: standardizeCode(x))
    
    # 筛选匹配的股票
    matched_df = df_processed[df_processed[code_col].isin(processed_targets)].copy()
    
    # 找出未找到的股票代码
    found_codes = matched_df[code_col].unique().tolist()
    not_found_codes = [code for code in processed_targets if code not in found_codes]
    
    if keep_type:
        matched_df = df.loc[matched_df.index].copy()
        
    return matched_df, not_found_codes


def checkCodeinDf(df, target_code,code_col=None):
    """
    检查target_code中的股票代码是否全都包含在df中
    输入参数：
        df：包含了多支股票交易信息的DataFrame
        target_code：要查询的股票code列表
        code_col:包含股票代码的列名
    返回值:
        List[str]：不在df中的股票代码code
    """
    # 找到列名
    code_col = findCodeCol(df, code_col)
    
    # 处理目标代码：标准化+去重
    standardized_targets = list({standardizeCode(code) for code in target_code if standardizeCode(code)})
    
    # 标准化df中的股票代码，统一格式后去重，减少对比量
    df_code_set = set(
        df[code_col].dropna().astype(str).apply(
            lambda x: standardizeCode(x)
        ).unique()
    )
    
    # 筛选为匹配到的股票代码
    not_found_codes = [code for code in standardized_targets if code not in df_code_set]
    return not_found_codes
    


def getCodeList(df, code_col=None):
    """
    获取df中的所有股票代码，去重后以列表的形式返回
    输入参数:
        df: 包含股票代码的dataframe
        code_col: 包含股票代码的列名(可不指定,会不断更新本函数来应对千奇百怪的列名)
    返回值：
        List[str]: 去重后标准化的6位股票代码列表（如['000001', '600001', '000002']）
    """
    # 找到列名
    code_col = findCodeColname(df, code_col)
    # 去重
    raw_codes = df[code_col].dropna().unique().tolist()

    # 标准化股票代码(统一转为6位纯数字字符串，兼容多格式)
    standardized_codes = []
    for code in raw_codes:
        code_str = standardizeCode(code)
        if code_str not in standardized_codes:
            standardized_codes.append(code_str)

    standardized_codes.sort()
    return standardized_codes
