import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 原始数据路径
root_path = '/data1/1shared/lijun/ecg/github_code/code-15/data/'
input_csv = root_path + 'exams.csv'
output_path = '/data1/1shared/yanmingke/code15/'

# 读取输入数据
data = pd.read_csv(input_csv)

# 将布尔值转换为 0 和 1
bool_columns = ['is_male', '1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']
data[bool_columns] = data[bool_columns].astype(int)

# 初始化目标 DataFrame
output_data = pd.DataFrame(columns=[
    "exam_id", "age", "is_male", "nn_predicted_age", "patient_id", "death", "timey", 
    "normal_ecg", "trace_file", "ecg", "1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"
])

# 遍历每行处理数据
for j in tqdm(range(data.shape[0])):
    exam_id = data['exam_id'][j]
    file_path = root_path + data['trace_file'][j]

    try:
        # 读取 HDF5 文件
        with h5py.File(file_path, "r") as f:
            ids = f['exam_id']
            tracings = f['tracings']

            # 查找匹配的 index
            i_temp = -1
            for i, id in enumerate(ids):
                if id == exam_id:
                    i_temp = i
                    break

            # 如果找到匹配的 ECG 数据
            if i_temp != -1:
                ecg_data = tracings[i_temp].tolist()
            else:
                ecg_data = []

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        ecg_data = []

    # 添加到目标 DataFrame
    output_data = pd.concat([
        output_data,
        pd.DataFrame({
            "exam_id": [data['exam_id'][j]],
            "age": [data['age'][j]],
            "is_male": [data['is_male'][j]],
            "nn_predicted_age": [data['nn_predicted_age'][j]],
            "patient_id": [data['patient_id'][j]],
            "death": [data['death'][j]],
            "timey": [data['timey'][j]],
            "normal_ecg": [data['normal_ecg'][j]],
            "trace_file": [data['trace_file'][j]],
            "ecg": [ecg_data],  # 将 ecg 数据存储为嵌套列表
            "1dAVb": [data["1dAVb"][j]],
            "RBBB": [data["RBBB"][j]],
            "LBBB": [data["LBBB"][j]],
            "SB": [data["SB"][j]],
            "ST": [data["ST"][j]],
            "AF": [data["AF"][j]]
        })
    ], ignore_index=True)

# 数据集划分
train_df, test_df = train_test_split(output_data, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# 重置索引
train_df.reset_index(inplace=True, drop=True)
val_df.reset_index(inplace=True, drop=True)
test_df.reset_index(inplace=True, drop=True)

# 保存划分后的数据集，命名为 code15_train, code15_val, code15_test
train_csv_path = output_path + 'code15_train.csv'
val_csv_path = output_path + 'code15_val.csv'
test_csv_path = output_path + 'code15_test.csv'

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

# 打印结果
print(f"训练集保存到: {train_csv_path}, shape: {train_df.shape}")
print(f"验证集保存到: {val_csv_path}, shape: {val_df.shape}")
print(f"测试集保存到: {test_csv_path}, shape: {test_df.shape}")