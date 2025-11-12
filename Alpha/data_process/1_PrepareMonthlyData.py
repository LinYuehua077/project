import pandas as pd
from tqdm import tqdm

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from func import SelectStock as ss
from func import ConcatDataByMonth as cdbm

prepare_data = False
merge_data = True

if prepare_data:
    # 设置数据路径
    path = "/home/vscode/workspace/data/store/external_history/MintueData/TickData/"
    price_path = os.path.join(path, 'Price')
    data_path = os.path.join(path, 'Data')
    price_files = ss.setTimeRange(price_path, 'parquet', 20240101, 20251107)
    data_files = ss.setTimeRange(data_path, 'parquet', 20240101, 20251107)

    # 检查两个列表的内容是否一致
    if not ss.areFileListsIdentical:
        raise ValueError('两个数据集的内容对不上，存在差异')

    # 生成从起始年月到截止年月之间的所有月份信息
    target_months = cdbm.generateMonthRange(2024, 1, 2025, 11)
    print(f"处理月份范围: {len(target_months)} 个月")
    print(f"从 {target_months[0]} 到 {target_months[-1]}")

    # 处理price_files数据
    file_path_lst = price_files
    print('正在处理price数据')
    for i, (year, month) in enumerate(target_months):
        print(f"\n处理第 {i+1}/{len(target_months)} 个周期: {year}年{month}月")

        # 判断是否需要包含上个月数据：除去第一个月，都要包含上个月的数据
        include_previous = (i > 0)
        # 判断是否需要包含下个月数据：除去最后一个月，都要包含下个月的数据
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
            output_file = f"../temp_data/Price/{year}{month:02d}.parquet"
            monthly_data.to_parquet(output_file)
            print(f"数据已保存: {output_file}")

            # 清理内存
            del monthly_data

    # 处理data_file数据
    file_path_lst = data_files
    print('正在处理data数据')
    for i, (year, month) in enumerate(target_months):
        print(f"\n处理第 {i+1}/{len(target_months)} 个周期: {year}年{month}月")

        # 判断是否需要包含上个月数据：不是第一个处理周期就需要
        include_previous = (i > 0)

        monthly_data = cdbm.processMonthlyData(
            file_path_lst=file_path_lst,
            year=year,
            month=month,
            data_type='data',
            include_previous_last_day=include_previous
        )

        if monthly_data is not None:
            # 保存结果，文件名包含年份信息
            output_file = f"../temp_data/Data/{year}{month:02d}.parquet"
            monthly_data.to_parquet(output_file)
            print(f"数据已保存: {output_file}")

            # 清理内存
            del monthly_data

if merge_data:
    data_lst = sorted(os.listdir("../temp_data/Data"))
    price_lst = sorted(os.listdir("../temp_data/Price"))
    for i in tqdm(range(len(data_lst))):
        if data_lst[i] != price_lst[i]:
            print(f"第{i}次循环出错，data[{i}]={data[i]}，price[{i}]={price[i]}")
        else:
            data_path = os.path.join("../temp_data/Data",data_lst[i])
            price_path = os.path.join("../temp_data/Price",price_lst[i])
            df_price = pd.read_parquet(data_path)
            df_data = pd.read_parquet(price_path)
            df_price_indexed = df_price.set_index(['code', 'DateTime'])
            df_data_indexed = df_data.set_index(['code', 'DateTime'])
            merged_df = df_price_indexed.join(df_data_indexed, how='left').reset_index()
            # 保存结果，文件名包含年份信息
            output_file = f"../temp_data/mergedData/{data_lst[i]}"
            merged_df.to_parquet(output_file)