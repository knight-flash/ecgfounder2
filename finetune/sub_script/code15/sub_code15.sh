task_name=$1
backbone=$2
pretrain_path=$3
ckpt_dir="/home/yanmingke/E-Zero/checkpoint/$task_name"

python /home/yanmingke/E-Zero/finetune/main_single_local_1.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset code15 \
    --pretrain_path $pretrain_path \
    --ratio 5 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 10 \
    --name $task_name

python /home/yanmingke/E-Zero/finetune/main_single_local_1.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset code15 \
    --pretrain_path $pretrain_path \
    --ratio 10 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 10 \
    --name $task_name

python /home/yanmingke/E-Zero/finetune/main_single_local_1.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset code15 \
    --pretrain_path $pretrain_path \
    --ratio 25 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 10 \
    --name $task_name

python /home/yanmingke/E-Zero/finetune/main_single_local_1.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset code15 \
    --pretrain_path $pretrain_path \
    --ratio 50 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 10 \
    --name $task_name

python /home/yanmingke/E-Zero/finetune/main_single_local_1.py \
    --checkpoint-dir $ckpt_dir \
    --batch-size 16 \
    --dataset code15 \
    --pretrain_path $pretrain_path \
    --ratio 100 \
    --learning-rate 0.001 \
    --backbone $backbone \
    --epochs 10 \
    --name $task_name