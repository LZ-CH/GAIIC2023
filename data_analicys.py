import csv
import numpy as np

with open('/data1/luojingzhou/datasets/gaiic_dataset/semi_train.csv', newline='') as csvfile:
    # 读取csv文件
    reader = csv.reader(csvfile)
    # 创建两个空列表，用于存储每一列的数据
    column1 = []
    column2 = []
    column3 = []
    # 遍历每一行数据，将每一列的数据添加到对应的列表中
    max_input_L = 0
    max_output_L = 0
    max_L = 0
    min_input_L = 10000
    min_output_L = 10000
    L = 0
    count = 0
    for row in reader:
        
        _, col1, col2, col3 = row
        a = []
        b = []
        c = []
        a.extend(map(int, col1.split()))
        b.extend(map(int, col2.split()))
        c.extend(map(int, col3.split()))
        if len(c)==0:
            count += 1
            print(col3=='')
    print(count)
#         L += len(list(a))+len(list(b))
#         if max_input_L<len(list(c)):
#             max_input_L = len(list(c))
#         # if max_output_L<len(list(b)):
#         #     max_output_L = len(list(b))
#         max_L = max(max_L,len(list(a))+len(list(c)))
#         if min_input_L>len(list(c)):
#             min_input_L = len(list(c))
#         # if min_output_L>len(list(b)):
#         #     min_output_L = len(list(b))
#         column1.extend(map(int, col1.split()))
#         column2.extend(map(int, col2.split()))
#         column3.extend(map(int, col3.split()))
# text = column1 + column2   +column3
# print(max_L,max_input_L,min_input_L)
# exit()
# with open('/data1/luojingzhou/datasets/gaiic_dataset/semi_train.csv', newline='') as csvfile:
#     # 读取csv文件
#     reader = csv.reader(csvfile)
#     # 创建两个空列表，用于存储每一列的数据
#     column1 = []
#     column2 = []
#     column3 = []
#     # 遍历每一行数据，将每一列的数据添加到对应的列表中
#     max_input_L = 0
#     max_output_L = 0
#     max_L = 0
#     min_input_L = 10000
#     min_output_L = 10000
#     L = 0
#     count = 0
#     for row in reader:
#         count += 1
#         _, col1, col2, col3 = row
#         a = []
#         b = []
#         c = []
#         a.extend(map(int, col1.split()))
#         b.extend(map(int, col2.split()))
#         c.extend(map(int, col3.split()))
#         L += len(list(a))+len(list(b))
#         # if max_input_L<len(list(a)):
#         #     max_input_L = len(list(a))
#         # if max_output_L<len(list(b)):
#         #     max_output_L = len(list(b))
#         # max_L = max(max_L,len(list(a))+len(list(b)))
#         # if min_input_L>len(list(a)):
#         #     min_input_L = len(list(a))
#         # if min_output_L>len(list(b)):
#         #     min_output_L = len(list(b))
#         column1.extend(map(int, col1.split()))
#         column2.extend(map(int, col2.split()))
#         column3.extend(map(int, col3.split()))
# text = column1 + column2   +column3

# # 打开csv文件
# with open('/data1/luojingzhou/datasets/gaiic_dataset/train.csv', newline='') as csvfile:
#     # 读取csv文件
#     reader = csv.reader(csvfile)
#     # 创建两个空列表，用于存储每一列的数据
#     column1 = []
#     column2 = []
#     # 遍历每一行数据，将每一列的数据添加到对应的列表中
#     max_input_L = 0
#     max_output_L = 0
#     max_L = 0
#     min_input_L = 10000
#     min_output_L = 10000
#     L = 0
#     count = 0
#     for row in reader:
#         count += 1
#         _, col1, col2 = row
#         a = []
#         b = []
#         a.extend(map(int, col1.split()))
#         b.extend(map(int, col2.split()))
#         L += len(list(a))+len(list(b))
#         if max_input_L<len(list(a)):
#             max_input_L = len(list(a))
#         if max_output_L<len(list(b)):
#             max_output_L = len(list(b))
#         max_L = max(max_L,len(list(a))+len(list(b)))
#         if min_input_L>len(list(a)):
#             min_input_L = len(list(a))
#         if min_output_L>len(list(b)):
#             min_output_L = len(list(b))
#         column1.extend(map(int, col1.split()))
#         column2.extend(map(int, col2.split()))
# text = column1 + column2   
with open('/data1/luojingzhou/datasets/gaiic_dataset/preliminary_a_test.csv', newline='') as csvfile:
    # 读取csv文件
    reader = csv.reader(csvfile)
    # 创建两个空列表，用于存储每一列的数据
    column1 = []

    # 遍历每一行数据，将每一列的数据添加到对应的列表中

    for row in reader:
        _, col1 = row
        a = []

        a.extend(map(int, col1.split()))

        column1.extend(map(int, col1.split()))
text +=  column1



# 使用numpy计算最大值、最小值和平均值
max_val = np.max(text)
min_val = np.min(text)
avg_val = np.mean(text)

# 使用numpy统计数字分布
hist, bins = np.histogram(text, bins=range(min_val, max_val+2))

for i in range(len(hist)):
    if hist[i]<10:
        print(bins[i],bins[i+1])
print('数字分布统计：')
for i in range(len(hist)):
    print(f'{bins[i]} - {bins[i+1]-1}: {hist[i]}')

# 打印最大值、最小值和平均值
print(max_val,min_val,avg_val)
print('max_input_L,max_output_L:',max_input_L,max_output_L)
print('min_input_L,min_output_L:',min_input_L,min_output_L)

#pred
with open('/data1/luojingzhou/projects/gaiic_hugging/pred.csv', newline='') as csvfile:
    # 读取csv文件
    reader = csv.reader(csvfile)
    # 创建两个空列表，用于存储每一列的数据
    column1 = []

    # 遍历每一行数据，将每一列的数据添加到对应的列表中

    for row in reader:
        _, col1 = row
        a = []

        a.extend(map(int, col1.split()))

        column1.extend(map(int, col1.split()))
text =  column1

# 使用numpy计算最大值、最小值和平均值
max_val = np.max(text)
min_val = np.min(text)
avg_val = np.mean(text)

# 使用numpy统计数字分布
hist, bins = np.histogram(text, bins=range(min_val, max_val+2))
print('数字分布统计：')
for i in range(len(hist)):
    print(f'{bins[i]} - {bins[i+1]-1}: {hist[i]}')

# 打印最大值、最小值和平均值
print('max_val,min_val,avg_val:',max_val,min_val,avg_val)
