# 项目结构一览

```
Alpha/
  ├── data_process/
  │    ├── 1_PrepareMonthlyData.py
  │    ├── 2_AdjData.py
  │    └── 3_CalculateFactor.py
  │
  ├── func/
  │    ├── AdjustStock.py
  │    ├── CalculateFactor.py
  │    ├── ConcatDataByMonth.py
  │    └── SelectStock.py
  │
  ├── log/
  │    └── adjustment_log.txt
  │
  ├── temp_data/
  │    ├── Data/
  │    ├── Price/
  │    ├── mergedData/
  │    ├── adjustedData/
  │    └── adjustment.parquet
  │
  ├── LightGBM/
  │    ├── 1_process_data.py
  │    ├── 2_main.py
  │    ├── model.py
  │    ├── tools.py
  │    └── results/
  │
  └── pegformer/
       ├── __init__.py
       ├── model/
       │    ├── __init__.py
       │    ├── patch_embedding.py
       │    ├── transformer_encoder.py
       │    ├── gru_decoder.py
       │    └── pegformer.py
       └── utils/
            ├── __init__.py
            ├── config.py
            └── model_utils.py
```



# 项目内容介绍

## func 文件夹

**为实现函数功能的文件，只提供各个函数功能的实现，不负责具体的任务流程**

> [!TIP]
>
> - **CalculateFactor.py**：计算股票因子的功能函数
>
> - **class MinuteFactorCalculator**：用于计算计算股票因子的类
>      - **calculate_factors()**：计算指定的因子(支持同时提交多个因子表达式来计算多个因子、支持一个df里有多支股票同时计算因子)
>      - **get_factors_by_stock(stock_code)**：获取指定股票的因子数据
>
>      - **def Alpha158_factor()**：用于创建Alpha158因子
>
>    - **ConcatDataByMonth.py**：用于按月份合并数据的功能函数
>    
>
> - **SelectStock.py**：用于灵活地按时间进行划分股票数据集的功能函数
> - **AdjustStock.py**：股票数据复权函数，包含优化后的实现方式



## data_process 文件夹

**为计算或处理数据集的脚本文件**

> [!IMPORTANT]
>
> - **0_CalculateAdjustmentFactor.py**：计算股票每日的复权因子的脚本
> - **1_PrepareMonthlyData.py**：先按年月日范围获划分定数据，再按月份合并数据的脚本
>   - 按时间划分数据时会灵活的匹配，若起始日期当天没有交易数据，则会往后（未来）自动寻找最近的交易日作为第一条数据，若截止日期当天没有交易数据，则会往前（过去）自动寻找最近的交易日作为最后一条数据，在日期选择上比较灵活，会自动修正无交易数据的情况
>   - 为了方便因子计算，每个月都会包含上一个月最后一次交易日的信息，例如7月份的交易数据里会包含6月份最后一天的交易数据，6月份的交易数据里会包含5月份最后一天的交易数据，从而既不用将数据合并成一个单一的大文件，也不会让跨窗口计算的因子因为月份而中断
> - **2_AdjData.py**：对准备好的、按月份处理过的分钟级交易数据进行复权操作的脚本，复权 ['open'、'close'、'high'、'low'、'volume'] 列
> - **3_selectStock.py**：按指数选择股票（会筛选掉存在交易数据缺失的股票代码）
> - **3_CalculateFactor.py**：计算 复权后的、指定的股票集合的、因子值的 脚本



## temp_data 文件夹（数据已忽略上传）

**存放计算、处理好的中间数据的文件夹**

> [!CAUTION]
>
> - **adjustment_temp.parquet**：将原始数据和复权后的数据整合在一起的临时文件
> - **adjustment.parquet**： 存放计算好的复权因子的文件
> - **factors.xlsx**：存放计算好的股票因子值的文件

### Data 子文件夹

> [!CAUTION]
>
> 存放按月份处理后的Data文件
>
> - **202507.parquet**
> - **202508.parquet**
> - **......**

### Price 子文件夹

> [!CAUTION]
>
> 存放按月份处理后的Price文件
>
> - **202507.parquet**
> - **202508.parquet**
> - **......**

### mergedData 子文件夹

> [!CAUTION]
>
> 存放按要求合并后的Data和Price数据的文件
>
> - **202507.parquet**
> - **202508.parquet**
> - **......**

### adjustedData 子文件夹

> [!CAUTION]
>
> 存放复权后的mergedData文件
>
> - **202507.parquet**
> - **202508.parquet**
> - **......**



## LightGBM 文件夹

> [!IMPORTANT]
>
> - **1_process_data.py**：给模型前，最后再处理一次数据，确保数据无误
> - **2_main.py**：训练模型、筛选前 n 个重要的因子
> - **model.py**：模型结构模块（用于调用）
> - **tools.py**：功能函数模块（用于调用）
> - **results 文件夹**：存放模型预测结果



## pegformer 文件夹

**存放pegformer模型相关的实现代码**

### utils 子文件夹

**存放模型的通用功能模块**

> [!NOTE]
>
> 

### model 子文件夹

**存放模型结构、训练、评估相关代码，是深度学习项目的核心逻辑层**

> [!NOTE]
>
> - **patch_embedding.py**：pegformer的embedding层，将分钟数据按照patch_size切分成多个片段，然后进行embedding
> - **transformer_encoder.py**：transformer的编码器模块，用于捕获时序数据之间存在的潜在关系
> - **gru_decode.py**：门控循环单元（GRU）解码器模块
> - **pegformer.py**：pegformer模型的模型结构
