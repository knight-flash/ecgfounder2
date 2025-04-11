import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, roc_curve, auc
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from finetune_dataset_code15 import getdataset
from models.resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from models.vit1d import vit_base, vit_small, vit_tiny, vit_middle
from models.net1d import Net1D_18

# ------------------- 日志配置 -------------------
import logging

logging.basicConfig(
    filename="test.log",     # 日志文件
    filemode="a",            # 追加模式
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO       # 记录 INFO 级别及以上的日志
)

# 记录 print() 输出
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)  # 终端显示
        self.log.write(message)       # 写入日志文件
        self.log.flush()              # 立即写入文件
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()




# ------------------- 参数配置 -------------------
parser = argparse.ArgumentParser(description='MERL Finetuning')
parser.add_argument('--dataset', default='ptbxl_super_class',
                    type=str, help='dataset name')
parser.add_argument('--ratio', default='100',
                    type=int, help='training data ratio')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--test-batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size for test/val')
parser.add_argument('--learning-rate', default=0.3, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--pretrain_path', default='/data1/1shared/lijun/ecg/E-Zero/checkpoints/resnet-18_new_preprocess_prmopt_0_ckpt.pth', type=str,
                    help='path to pretrain weight directory')
parser.add_argument('--checkpoint-dir', default='./checkpoint_finetune/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--backbone', default='resnet18', type=str, metavar='B',
                    help='backbone name')
parser.add_argument('--num_leads', default=12, type=int, metavar='B',
                    help='number of leads')
parser.add_argument('--name', default='LinearProbing', type=str, metavar='B',
                    help='exp name')
parser.add_argument('--local_rank', default=0, type=int, help='Used for DDP')  
# ↑ 如果使用 torchrun ，PyTorch 会自动注入 local_rank 到脚本。手动多卡时需要自己传这个值。


def main():
    # ------------------- 1. 初始化分布式环境 -------------------
    args = parser.parse_args()
    dist.init_process_group(backend="nccl")
    torch.cuda.empty_cache()

    # 获取本进程的 rank（全局进程编号）和可用 GPU 数
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count() # 如果每个进程只使用单卡，则 local_rank 与 rank 一般对应
    
    # 设置当前进程使用哪张 GPU
    torch.cuda.set_device(device_id)

    # 只有 rank=0 时才进行部分打印，避免多卡重复输出
    if rank == 0:
        print(f"Start running DDP on rank {rank}, total world_size = {world_size}.")

    # ------------------- 2. 准备数据集和 DataLoader -------------------
    batch_size = int(args.batch_size)
    test_batch_size = int(args.test_batch_size)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.checkpoint_dir / "test.log"
    dataset_dir = args.checkpoint_dir / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(str(log_file) ) 
    sys.stderr = sys.stdout  # 让错误信息也写入日志
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True

    if rank == 0:
        print(f'this task use {args.dataset} dataset')

    data_split_path = './data_split'
    data_meta_path = '/data1/1shared/jinjiarui/run/ECG-LLM-main/Datasets'
    
    # 根据不同数据集进行加载示例，这里保留原逻辑
    if 'ptbxl' in args.dataset:
        data_path = f'/data1/1shared/yanmingke/ptbxl/physionet.org/files/ptb-xl/1.0.3'
        data_split_path = os.path.join(data_split_path, f'ptbxl/{args.dataset[6:]}')
        
        train_csv_path = os.path.join(data_split_path, f'{args.dataset}_train.csv')
        val_csv_path = os.path.join(data_split_path, f'{args.dataset}_val.csv')
        test_csv_path = os.path.join(data_split_path, f'{args.dataset}_test.csv')
        
        train_dataset = getdataset(data_path, train_csv_path, mode='train', dataset_name='ptbxl', ratio=args.ratio,
                                   backbone=args.backbone)
        val_dataset = getdataset(data_path, val_csv_path, mode='val', dataset_name='ptbxl',
                                   backbone=args.backbone)
        test_dataset = getdataset(data_path, test_csv_path, mode='test', dataset_name='ptbxl',
                                   backbone=args.backbone)

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    elif args.dataset == 'CPSC2018':
        data_path = f'{data_meta_path}/icbeb2018/records500'
        data_split_path = os.path.join(data_split_path, args.dataset)
        
        train_csv_path = os.path.join(data_split_path, f'{args.dataset}_train.csv')
        val_csv_path = os.path.join(data_split_path, f'{args.dataset}_val.csv')
        test_csv_path = os.path.join(data_split_path, f'{args.dataset}_test.csv')
        
        train_dataset = getdataset(data_path, train_csv_path, mode='train', dataset_name='icbeb', ratio=args.ratio,
                                   backbone=args.backbone)
        val_dataset = getdataset(data_path, val_csv_path, mode='val', dataset_name='icbeb',
                                   backbone=args.backbone)
        test_dataset = getdataset(data_path, test_csv_path, mode='test', dataset_name='icbeb',
                                   backbone=args.backbone)

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    elif args.dataset == 'CSN':
        data_path = f'{data_meta_path}/downstream/'
        data_split_path = os.path.join(data_split_path, args.dataset)
        
        train_csv_path = os.path.join(data_split_path, f'{args.dataset}_train.csv')
        val_csv_path = os.path.join(data_split_path, f'{args.dataset}_val.csv')
        test_csv_path = os.path.join(data_split_path, f'{args.dataset}_test.csv')
        
        train_dataset = getdataset(data_path, train_csv_path, mode='train', dataset_name='chapman', ratio=args.ratio,
                                   backbone=args.backbone)
        val_dataset = getdataset(data_path, val_csv_path, mode='val', dataset_name='chapman',
                                   backbone=args.backbone)
        test_dataset = getdataset(data_path, test_csv_path, mode='test', dataset_name='chapman',
                                   backbone=args.backbone)

    elif args.dataset == 'code15':
        data_path = f'/data1/1shared/lijun/ecg/github_code/code-15/data/'
        data_split_path = os.path.join(data_split_path, args.dataset)

        train_csv_path = os.path.join(data_split_path, f'{args.dataset}_train.csv')
        val_csv_path = os.path.join(data_split_path, f'{args.dataset}_val.csv')
        test_csv_path = os.path.join(data_split_path, f'{args.dataset}_test.csv')
        
        train_dataset = getdataset(data_path, train_csv_path, mode='train', dataset_name='code15', ratio=args.ratio,
                                   backbone=args.backbone)
        val_dataset = getdataset(data_path, val_csv_path, mode='val', dataset_name='code15',
                                   backbone=args.backbone)
        test_dataset = getdataset(data_path, test_csv_path, mode='test', dataset_name='code15',
                                   backbone=args.backbone)

        args.labels_name = train_dataset.labels_name
        num_classes = 6

    # ---- 使用 DistributedSampler，让每个进程只处理一部分数据 ----
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=args.workers, 
        pin_memory=True
    )

    # 验证和测试可以选择只在 rank=0 上单卡跑，也可以多卡分布式跑，这里给出多卡一致的写法
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        sampler=val_sampler,
        num_workers=args.workers, 
        pin_memory=True
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_sampler,
        num_workers=args.workers, 
        pin_memory=True
    )

    # ------------------- 3. 构建模型并加载预训练权重 -------------------
    ckpt_path = args.pretrain_path
    # 加载预训练模型（注意，如果是自己保存的模型，需要根据 key 来调整）
    # 下面以 strict=False 为例，如果你的字典里是 "model_state_dict" 则修改读取方式
    # weights_only=True 是常见的自定义字段，如有需要可自行移除
    try:
        checkpoint = torch.load(ckpt_path, map_location=f'cuda:{device_id}')
        # 如果是 {"model_state_dict":..., ...} 这种格式：
        # state_dict = checkpoint["model_state_dict"]
        # model.load_state_dict(state_dict, strict=False)
    except:
        checkpoint = None

    if 'resnet' in args.backbone:
        if args.backbone == 'resnet18':
            model = ResNet18(num_classes=num_classes)
        elif args.backbone == 'resnet50':
            model = ResNet50(num_classes=num_classes)
        elif args.backbone == 'resnet101':
            model = ResNet101(num_classes=num_classes)
        
        if checkpoint is not None:
            # 如果直接是权重字典，可能是这样的方式：
            model.load_state_dict(checkpoint, strict=False)
        if rank == 0:
            print(f'load pretrained model from {args.pretrain_path}, backbone={args.backbone}, leads={args.num_leads}')
        
        # 若是线性探针，冻结除最后线性层外的参数
        
        for param in model.parameters():
                param.requires_grad = False
        for param in model.linear.parameters():
                param.requires_grad = True
    
    # 在 linear 层后加上 Sigmoid 激活
            # model.linear = nn.Sequential(
            # model.linear,
            # nn.Sigmoid()
            #                             )
        if rank == 0:
                print(f'freeze backbone for {args.name} with {args.backbone} and added Sigmoid activation')


    elif 'vit' in args.backbone:
        if args.backbone == 'vit_tiny':
            model = vit_tiny(num_classes=num_classes, num_leads=args.num_leads)
        elif args.backbone == 'vit_small':
            model = vit_small(num_classes=num_classes, num_leads=args.num_leads)
        elif args.backbone == 'vit_middle':
            model = vit_middle(num_classes=num_classes, num_leads=args.num_leads)
        elif args.backbone == 'vit_base':
            model = vit_base(num_classes=num_classes, num_leads=args.num_leads)
        
        if checkpoint is not None:
            model.load_state_dict(checkpoint, strict=False)
        
        if rank == 0:
            print(f'load pretrained model from {args.pretrain_path}, backbone={args.backbone}, leads={args.num_leads}')
        
        if 'linear' in args.name:
            for param in model.parameters():
                param.requires_grad = False
            model.reset_head(num_classes=num_classes)
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
            if rank == 0:
                print(f'freeze backbone for {args.name} with {args.backbone}')

    elif 'CLIP_ResNet' in args.backbone:
        if args.backbone == 'CLIP_ResNet':
            model = Net1D_18(num_classes=num_classes)
        
        ckpt = torch.load(ckpt_path, map_location='cpu')

        ckpt_filtered = {k: v for k, v in ckpt.items() if not k.startswith('dense')}
        
        model.load_state_dict(ckpt_filtered, strict=False)
        
        if rank == 0:
            print(f'load pretrained model from {args.pretrain_path}, backbone={args.backbone}, leads={args.num_leads}')
            
        if 'linear' in args.name:
            for param in model.parameters():
                param.requires_grad = False #默认False
            if rank == 0:
                    print(f'freeze backbone for {args.name} with {args.backbone}')
            for param in model.dense.parameters():
                param.requires_grad = True
           

    # ------------------- 4. 将模型放到 GPU 并使用 DDP 包装 -------------------
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], output_device=device_id,find_unused_parameters=True)
    
    # ------------------- 5. 定义优化器、学习率策略、损失函数等 -------------------
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1, last_epoch=-1)
    criterion = nn.BCEWithLogitsLoss()

    # 注意：因为模型被 DDP 包裹，若要保存/加载，需要用 model.module 来获取实际的子模块参数
    checkpoint_path = os.path.join('../checkpoint/linear', f"{args.backbone}-checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        os.makedirs(str(dataset_dir), exist_ok=True)
        start_epoch = 0



    global_step = 0

    log = {
        'epoch': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_auc': [],
        'test_acc': [],
        'test_f1': [],
        'test_precision': [],
        'test_recall': [],
        'test_auc': []
    }
    class_log = {
        'val_log': [],
        'test_log': []
    }
    
    scaler = GradScaler()
    for epoch in tqdm(range(start_epoch, args.epochs)):
        model.train()
        for step, (ecg, target) in tqdm(enumerate(train_loader, start=epoch * len(train_loader))):
            optimizer.zero_grad()
            with autocast():
                output = model(ecg.to(f'cuda:{device_id}'))
                loss = criterion(output, target.to(f'cuda:{device_id}'))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        val_acc, val_f1, val_precision, val_recall, val_auc, val_metric_class = infer(model, val_loader, args,device_id)
        test_acc, test_f1, test_precision, test_recall, test_auc, test_metric_class = infer(model, test_loader, args,device_id)

        log['epoch'].append(epoch)
        log['val_acc'].append(val_acc)
        log['val_f1'].append(val_f1)
        log['val_precision'].append(val_precision)
        log['val_recall'].append(val_recall)
        log['val_auc'].append(val_auc)
        log['test_acc'].append(test_acc)
        log['test_f1'].append(test_f1)
        log['test_precision'].append(test_precision)
        log['test_recall'].append(test_recall)
        log['test_auc'].append(test_auc)

        class_log['val_log'].append(val_metric_class)
        class_log['test_log'].append(test_metric_class)

        scheduler.step()
    
    csv = pd.DataFrame(log)
    csv.columns = ['epoch', 'val_acc',
                    'val_f1', 'val_precision',
                      'val_recall', 'val_auc', 
                      'test_acc',
                        'test_f1', 'test_precision',
                          'test_recall', 'test_auc']
    
    val_class_csv = pd.concat(class_log['val_log'], axis=0)
    test_class_csv = pd.concat(class_log['test_log'], axis=0)
    val_class_csv.to_csv(f'{args.checkpoint_dir}/'+args.name+'-'+args.backbone+'-B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'-val-class.csv', index=False)
    test_class_csv.to_csv(f'{args.checkpoint_dir}/'+args.name+'-'+args.backbone+'-B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'-test-class.csv', index=False)

    csv.to_csv(f'{args.checkpoint_dir}/'+args.name+'-'+args.backbone+'-B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'.csv', index=False)
    
    print(f'max val acc: {max(log["val_acc"])}\n \
            max val f1: {max(log["val_f1"])}\n \
            max val precision: {max(log["val_precision"])}\n \
            max val recall: {max(log["val_recall"])}\n \
            max val auc: {max(log["val_auc"])}\n \
            max test acc: {max(log["test_acc"])}\n \
            max test f1: {max(log["test_f1"])}\n \
            max test precision: {max(log["test_precision"])}\n \
            max test recall: {max(log["test_recall"])}\n \
                max test auc: {max(log["test_auc"])}\n')
    # plot each metric in one subplot
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.plot(log['epoch'], log['val_acc'], label='val_acc')
    plt.plot(log['epoch'], log['test_acc'], label='test_acc')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(log['epoch'], log['val_f1'], label='val_f1')
    plt.plot(log['epoch'], log['test_f1'], label='test_f1')
    plt.legend()
    plt.subplot(2, 2, 3)
    # since we donot compute precision and recall in there. so this figure is not useful.
    # plt.plot(log['epoch'], log['val_precision'], label='val_precision')
    # plt.plot(log['epoch'], log['test_precision'], label='test_precision')
    # plt.plot(log['epoch'], log['val_ecall'], label='val_recall')
    # plt.plot(log['epoch'], log['test_recall'], label='test_recall')
    # plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(log['epoch'], log['val_auc'], label='val_auc')
    plt.plot(log['epoch'], log['test_auc'], label='test_auc')
    plt.legend()
    plt.savefig(f'{args.checkpoint_dir}/'+args.name+'-'+args.backbone+'-B-'+str(batch_size)+args.dataset+'R-'+str(args.ratio)+'.png')
    plt.close()

@torch.no_grad()
def infer(model, loader, args,device_id):
    """适用于多卡训练的评估函数"""
    model.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # 各进程本地存储
    pred_buffer, true_buffer = [], []

    # 使用异步数据加载
    with torch.cuda.stream(torch.cuda.Stream(device=device_id)):
        for ecg, target in tqdm(loader, disable=rank!=0):
            ecg = ecg.to(device_id, non_blocking=True)
            target = target.to(device_id, non_blocking=True)
            
            with torch.no_grad():
                pred = model(ecg)
            
            pred_buffer.append(pred)
            true_buffer.append(target)

    # 合并当前进程数据
    all_pred = torch.cat(pred_buffer)
    all_true = torch.cat(true_buffer)

    # 跨卡同步数据
    if world_size > 1:
        # 收集各卡数据尺寸
        size_tensor = torch.tensor([all_pred.shape[0]], device=device_id)
        sizes = [torch.empty_like(size_tensor) for _ in range(world_size)]
        dist.all_gather(sizes, size_tensor)

        # 准备接收缓冲区
        max_size = max(sizes)[0].item()
        padded_pred = torch.zeros(max_size, all_pred.shape[1], 
                                device=device_id, dtype=all_pred.dtype)
        padded_true = torch.zeros(max_size, all_true.shape[1],
                                device=device_id, dtype=all_true.dtype)
        padded_pred[:all_pred.shape[0]] = all_pred
        padded_true[:all_true.shape[0]] = all_true

        # 收集所有数据
        gathered_pred = [torch.zeros_like(padded_pred) for _ in range(world_size)]
        gathered_true = [torch.zeros_like(padded_true) for _ in range(world_size)]
        dist.all_gather(gathered_pred, padded_pred)
        dist.all_gather(gathered_true, padded_true)

        # 合并有效数据
        valid_preds = []
        valid_trues = []
        for i in range(world_size):
            valid_len = sizes[i][0].item()
            valid_preds.append(gathered_pred[i][:valid_len])
            valid_trues.append(gathered_true[i][:valid_len])
        
        all_pred = torch.cat(valid_preds).cpu().numpy()
        all_true = torch.cat(valid_trues).cpu().numpy()
    else:
        all_pred = all_pred.cpu().numpy()
        all_true = all_true.cpu().numpy()

    # 仅主进程计算指标
    if rank != 0:
        return None, None, None, None, None, None

    # 安全计算AUC
    def safe_auc(y_true, y_pred):

        
        try:
            return roc_auc_score(y_true, y_pred)
        except Exception as e:
            print(f"AUC calculation failed: {str(e)}")
            return 0.5

    # 初始化指标存储
    auc_scores = []
    f1_scores = []
    acc_scores = []
    metric_dict = {name: [] for name in args.labels_name}

    # 逐类别计算
    for col in range(all_true.shape[1]):
        y_true = all_true[:, col]
        y_pred = all_pred[:, col]
        y_true = (y_true >= 0.5).astype(int)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)
        # AUC
        auc = safe_auc(y_true, y_pred)
        auc_scores.append(auc)
        metric_dict[args.labels_name[col]].append(auc)

        # F1与准确率
        prec, rec, threshs = precision_recall_curve(y_true, y_pred)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1)
        best_f1 = f1[best_idx]
        best_thresh = threshs[best_idx]

        f1_scores.append(best_f1)
        acc_scores.append(accuracy_score(y_true, y_pred >= best_thresh))

    # 计算宏平均
    macro_auc = np.mean(auc_scores)
    macro_f1 = np.mean(f1_scores) * 100
    macro_acc = np.mean(acc_scores) * 100

    return (
        macro_acc,
        macro_f1,
        0,  # precision暂不计算
        0,  # recall暂不计算
        macro_auc,
        pd.DataFrame(metric_dict)
    )


if __name__ == '__main__':
    main()