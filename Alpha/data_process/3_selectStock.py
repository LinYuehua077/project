import pandas as pd
from tqdm import tqdm

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)

from func import SelectStock as ss
from func import ConcatDataByMonth as cdbm

# 测试创业板指中的股票
df1 = pd.read_excel('../temp_data/20240105中证500.xls')
target_code = ss.getCodeList(df1)


print(f'起始状态共选择股票{len(target_code)}支...')
# 检查哪些股票存在数据缺失
loss_code = set()

# 获取处理好的因子文件
file_lst = sorted(os.listdir('../temp_data/adjustedData'))
for file in tqdm(file_lst, desc='处理缺失'):
    file_path = os.path.join("../temp_data/adjustedData", file)
    df = pd.read_parquet(file_path)              # 读取存放因子数据的文件
    target_code, temp_code = ss.checkCodeinDf(df, target_code) # 检查哪支股票存在数据缺失
    print(f'{file}缺失:{temp_code}, 数量{len(temp_code)}')
    loss_code.update(temp_code)

loss_code = sorted(list(loss_code))
print('缺失情况处理完毕')
print(f'共{len(loss_code)}支股票存在数据缺失')
print(f'剩余{len(target_code)}支股票数据齐全')

import json
with open("../log/target_code.json", "w", encoding="utf-8") as f:
    json.dump(target_code, f, ensure_ascii=False, indent=2)
with open("../log/loss_code.json", "w", encoding="utf-8") as f:
    json.dump(loss_code, f, ensure_ascii=False, indent=2)