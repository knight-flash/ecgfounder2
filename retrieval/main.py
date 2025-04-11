import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import tempfile
import os
from torch import optim
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import yaml
import sys
sys.path.append("../utils")
from utils_trainer import test_wBert
import utils_dataset_heedb as utils_dataset
import utils_builder

import wandb
import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import logging

# 配置日志
logging.basicConfig(
    filename="test.log",  # 日志文件
    filemode="a",          # 追加模式
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO      # 记录 INFO 级别及以上的日志
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

sys.stdout = Logger("test.log")  # 把 stdout 重定向到日志文件
sys.stderr = sys.stdout  # 让错误信息也写入日志



def ddp_main():
    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    # set up
    config = yaml.load(open("retrieval_config.yaml", "r"), Loader=yaml.FullLoader)
    if device_id == 0:
        run = wandb.init(
            # Set the project where this run will be logged
            project="Retrieval",
            name = config['wandb_name'],
            # Track hyperparameters and run metadata
            config={
                    "learning_rate": config['optimizer']['params']['lr'],
                    'weight_decay': config['optimizer']['params']['weight_decay'],
                    'ecg_model': config['network']['ecg_model'],
                    'batch_size': config['tester']['batch_size'],
                    'ckpt_path': config['network']['ckpt_path'],
                    'model_name': config['network']['model_name']
                    
            }
        )
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    # loading data path

    # define image-text dataset
    # dataset = utils_dataset.ECG_TEXT_Dsataset(
    #     data_path=data_path, dataset_name=config['dataset']['dataset_name'])
    # train_dataset = dataset.get_dataset(train_test='train')
    # val_dataset = dataset.get_dataset(train_test='val')
    test_dataset = utils_dataset.test_HEEDB_Dataset()
    #val_dataset = utils_dataset.val_MIMIC_Dataset()
    # building model part
    # --------------------
    ckpt_path = os.path.join(config['network']['ckpt_path'])
    if config['network']['model_name'] == 'ECGCLIP':
        model = utils_builder.ECGCLIP(config['network'])
    elif config['network']['model_name'] == 'AnotherModel':  # 假设另一个模型名称为'AnotherModel'
        model = utils_builder.AnotherModel(config['network'])  # 假设另一个模型的构建函数为AnotherModel
    else:
        raise ValueError(f"Unknown ECG model: {config['network']['model_name']}")    
    model = torch.nn.DataParallel(model,device_ids=[device_id])
    # 2. 加载模型权重
    checkpoint = torch.load(ckpt_path, map_location=f'cuda:{device_id}', weights_only=True)  # 如果使用 GPU，可将 map_location 设置为 'cuda'
    #print(checkpoint)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device_id)
    # 3. 切换模型到评估模式
    model.eval()
    '''
    # you can freeze bert from last layer to first layer.
    # set num of layer in config.yaml
    # default is freeze 9 layers
    # '''
    # if config['network']['free_layers'] is not None:
    #     for layer_idx in range(int(config['network']['free_layers'])):
    #         for param in list(model.lm_model.h[layer_idx].parameters()):
    #             param.requires_grad = False

    # model = model.to(device_id)
    # model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # --------------------

    # --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config['optimizer']['params'],
        betas=(0.9, 0.999)
    )

    # ---------xw-----------
    tester = test_wBert(model=model,
                            optimizer=optimizer,
                            device=rank,
                            model_name=config['wandb_name'],
                            **config['tester'])
    # --------------------
    
    # --------------------
    # I_T_P_tester
    tester.test(test_dataset)


ddp_main()
