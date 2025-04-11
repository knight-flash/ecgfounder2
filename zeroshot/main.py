import os
import random
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import sys
sys.path.append("../utils")
import utils_builder
from zeroshot_val import zeroshot_eval
import json
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import csv

def save_per_class_results(set_name, metrics, output_dir, per_class_metrics=None):
    if per_class_metrics is None:
        per_class_metrics = []
    
    results = {
        'set_name': set_name,
        'f1': metrics[0].item(),
        'acc': metrics[1].item(),
        'auc': metrics[2].item(),
        'per_class_metrics': per_class_metrics
    }
    output_path = os.path.join(output_dir, f"{set_name}_results.csv")
    print(output_path)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['set_name', 'f1', 'acc', 'auc'] + [f'class_{i}' for i in range(len(per_class_metrics[0]))])
        writer.writerow([results['set_name'], results['f1'], results['acc'], results['auc']] + per_class_metrics[0])
        for class_metrics in per_class_metrics[1:]:
            writer.writerow([''] + [''] * 4 + class_metrics)
def ddp_main():
    # 初始化分布式环境
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    # 加载配置
    config = yaml.load(open("zeroshot_config.yaml", "r"), Loader=yaml.FullLoader)
    
    # 设置随机种子（需保证不同进程不同种子）
    seed = config.get('seed', 42) + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 构建模型
    model_name = config.get('model_name', 'ECGCLIP')
    if model_name == 'ECGCLIP':
        model = utils_builder.ECGCLIP(config['network'])
    elif model_name == 'OtherModel':
        model = utils_builder.OtherModel(config['network'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")    
    # 加载checkpoint
    ckpt_path = os.path.join(config['network']['ckpt_path'])
    ckpt = torch.load(ckpt_path, map_location=f'cuda:{rank}')
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    # 包装为DDP模型
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 分布式评估参数
    args_zeroshot_eval = config['zeroshot']
    
    # 主进程打印信息
    if rank == 0:
        print(f"Starting evaluation on {world_size} GPUs...")

    # 同步所有进程
    dist.barrier()
    output_dir = os.path.join("../checkpoint/zeroshot")
    os.makedirs(output_dir, exist_ok=True)
    # 执行评估
    
    total_metrics = torch.zeros(3, device=rank)  # 存储f1, acc, auc
    
    for set_name in args_zeroshot_eval['test_sets']:
        f1, acc, auc, *_ = zeroshot_eval(
            model=ddp_model.module,  # 使用原始模型
            set_name=set_name,
            device=rank,
            args_zeroshot_eval=args_zeroshot_eval
        )
        save_per_class_results(set_name, torch.tensor([f1, acc, auc], device=rank), output_dir)
        # 累加指标
        total_metrics += torch.tensor([f1, acc, auc], device=rank)
    
    # 计算平均指标
    num_sets = len(args_zeroshot_eval['test_sets'])
    avg_metrics = total_metrics / num_sets
    
    # 汇总所有进程的结果
    dist.all_reduce(avg_metrics, op=dist.ReduceOp.SUM)
    avg_metrics /= world_size

    # 主进程打印结果
    if rank == 0:
        print(f"\nFinal Metrics (averaged across {world_size} GPUs):")
        print(f"Avg F1: {avg_metrics[0].item():.4f}")
        print(f"Avg Acc: {avg_metrics[1].item():.4f}")
        print(f"Avg AUC: {avg_metrics[2].item():.4f}")
    


    # 清理进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    # 启动分布式训练
    ddp_main()

