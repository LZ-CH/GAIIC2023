import pandas as pd
import numpy as np

# 读取csv文件
df = pd.read_csv('./gaiic_dataset/semi_train.csv',header=None)

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 计算训练集和验证集的样本数量
train_size = int(0.95 * len(df))
valid_size = len(df) - train_size

# 划分训练集和验证集
train_df = df[:train_size]
valid_df = df[train_size:]

# 输出训练集和验证集的样本数量
print(f'Train samples: {len(train_df)}')
print(f'Valid samples: {len(valid_df)}')

# 保存训练集和验证集
train_df.to_csv('./gaiic_dataset/semi_train_split.csv', index=False,header=None)
valid_df.to_csv('./gaiic_dataset/semi_valid_split.csv', index=False,header=None)
