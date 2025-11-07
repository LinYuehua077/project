# 项目内容介绍

## func 文件夹

为实现函数功能的文件，只提供各个函数功能的实现，不负责具体的任务流程

> [!TIP]
>
> 1. `CalculateFactor.py`：计算股票因子的功能函数
>
>    - **class MinuteFactorCalculator**：用于计算计算股票因子的类
>      - **calculate_factors()**：计算指定的因子(支持同时提交多个因子表达式来计算多个因子、支持一个df里有多支股票同时计算因子)
>      - **get_factors_by_stock(stock_code)**：获取指定股票的因子数据
>    - **def Alpha158_factor()**：用于创建Alpha158因子
>      - 输入参数：无
>      - 返回参数1： (列表) Alpha158因子的计算表达式(与参数2是一一对应关系)
>      - 返回参数2：(列表) Alpha158因子的因子名称(与参数1是一一对应关系)
>
>    
>
> 2. `SelectStock.py`：用于选取指定的股票数据
>
>    - **def setTimeRange()**: 用于筛选某个时间段内的交易数据
>      - 输入参数：
>        - file_path：存放所有时间的股票数据的文件夹地址
>        - file_type：文件的类型(例如parquet,h5,xls,xlsx),可以传递列表,比如['xls', 'xlsx']
>        - start: 开始日期, 若开始日期没有交易记录, 则往后(未来)自动延申
>        - end:   结束日期, 若结束日期没有交易记录, 则往前(过去)自动延伸
>      - 返回参数: (列表) 按时间升序排列的选中交易记录文件路径列表
>    - **def areFileListsIdentical(lst1, lst2)**：两个列表分别存放了某一个时间段的不同交易数据，本函数用于判断两个不同的交易数据在时间上是否一致，会不会出现lst1有某一天的数据但lst2却没有收录进来的错误情况
>      - 输入参数:
>        - lst1:存放第一种交易数据的列表
>        - lst2:存放第二种交易数据的列表
>      - 返回参数: (布尔值) True:二者相同、False:二者存在不同
>    - **def selectByCode(df, target_codes, code_col='code')**：根据目标股票代码列表筛选DataFrame，兼容多种股票代码格式，返回匹配结果和未找到的代码
>      - 输入参数:
>        - df: 原始股票数据DataFrame，需包含股票代码列
>        - target_codes: 目标股票代码列表，支持多种格式：
>          - 带市场后缀：['000001.SZ', '000002.SH']
>          - 纯数字字符串：['000001', '000002']、['1', '2']
>          - 整数：\[1, 2, 000001\]（注意：整数1会被转为'000001'补全6位）
>      - 返回参数: (元组) (匹配目标代码的DataFrame, 未找到的股票代码列表)
>
> 3. 



## data_process 文件夹

为计算或处理数据集的脚本文件

> [!IMPORTANT]
>
> - **CalculateAdjustmentFactor.py**：计算复权因子的脚本
> - **CalculateFactor.py**：计算数据集的因子值的脚本



## temp_data 文件夹

存放计算、处理好的中间数据的文件夹

> [!CAUTION]
>
> - **adjustment_temp.parquet**：将原始数据和复权后的数据整合在一起的临时文件
> - **adjsutment.parquet**： 存放计算好的复权因子的文件
> - **factors.xlsx**：存放计算好的股票因子值的文件

