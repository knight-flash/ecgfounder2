# ecgfounder2

# 使用指南

本实验暂时有四个实验，所有实验均使用launch.sh运行


     bash launch.sh <任务目录> <GPU数量> [其他python参数...]

所有实验结果都会放在checkpoint对应文件夹中，预训练模型会存在checkpoint下

## pretrain

目前暂定模型为ecgfounder作为ecg-encoder，MedCPT为text-encoder

其余的ecg-encoder：ResNet， ViT

其余的text-encoder：GPT2

（需要更多同行对比）

## retrieval

使用模型对数据集的ECG-TEXT和TEXT-ECG进行配对

以R@1 R@5 R@10 为指标，目前batch为1000

目前数据集：HEEDB

目前对比模型：Random

（需要更多同行对比）

## zeroshot

评估模型在未见诊断类别上的泛化能力，实现无需专门标注数据的零样本分类

以各个类的ACC，F1，AUC为指标

目前数据集：PTB-XL， CODE15

目前对比模型：MERL(待复现)

## finetune

线性检测实验， 检验ECG编码器提取的特征的判别能力

以各个类的ACC，F1，AUC为指标


目前数据集：PTB-XL， CODE15

目前对比模型：MERL(待复现)

问题：
1.需要能复现的同行对比（ecg和text）
2.ecgfounder有点大，训练的batch为32，可能会影响大batch下的对齐
3.仍有一些地址没改
