task_name=$1
backbone=$2
pretrain_path=$3
ckpt_dir="/home/yanmingke/E-Zero/checkpoint/$task_name"
main_script="/home/yanmingke/E-Zero/finetune/main_single_local_3.py"
ratios=(1 10 100)

# 遍历不同 ratio，多次启动训练
for r in "${ratios[@]}"; do
    echo "======= Start training with ratio=$r ======="

    # --nproc_per_node=4 表示单机使用4个GPU，如果你有2张卡，就改成 --nproc_per_node=2
    # --master_port=29500 是默认端口，也可以改成其他未被占用的端口
    torchrun --nproc_per_node=4  \
        $main_script \
        --checkpoint-dir "$ckpt_dir" \
        --batch-size 16 \
        --dataset ptbxl_rhythm \
        --pretrain_path "$pretrain_path" \
        --ratio $r \
        --learning-rate 0.001 \
        --backbone "$backbone" \
        --epochs 100 \
        --name "$task_name"
done



