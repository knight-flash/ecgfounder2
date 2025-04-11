#!/bin/bash

# 通用分布式训练启动脚本
# 用法: ./launch_ddp.sh <任务目录> <GPU数量> [其他python参数...]
# 示例:
#   ./launch_ddp.sh pretrain 4 --batch_size=256
#   ./launch_ddp.sh zeroshot 2 --lr=1e-5
#   ./launch_ddp.sh retrieval 8 --topk=100

# 参数检查

if [ $# -lt 2 ]; then
    echo "错误：参数不足"
    echo "用法: $0 <任务目录> <GPU数量> [其他参数...]"
    exit 1
fi

TASK_DIR=$1  # 任务目录名（pretrain/zeroshot/retrieval）
NP=$2        # GPU数量
shift 2      # 移除前两个参数，剩余参数传递给python

# 进入任务目录
if [ ! -d "$TASK_DIR" ]; then
    echo "错误：任务目录 '$TASK_DIR' 不存在"
    exit 1
fi
cd $TASK_DIR
if [ "$TASK_DIR" = "finetune" ]; then
    echo "启动finetune任务: $TASK_DIR"
    bash run_all_linear_mult.sh
    exit 0
fi

# 分布式参数配置
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}     # 默认主节点地址
MASTER_PORT=${MASTER_PORT:-29500}         # 默认起始端口
MASTER_PORT=$((MASTER_PORT + RANDOM % 100))  # 随机端口避免冲突

# 执行分布式训练命令
echo "启动DDP任务: $TASK_DIR"
echo "GPU数量: $NP | 主节点: $MASTER_ADDR:$MASTER_PORT"
echo "附加参数: $@"

torchrun \
    --nproc_per_node=$NP \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py "$@"

# 返回上级目录（可选）
cd ..
