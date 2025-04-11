taskname='linear_probe_ptbxl_0310_mult_1343'
backbone='CLIP_ResNet'
pretrain_path='/home/yanmingke/E-Zero/checkpoint_18000.pt'

# taskname='linear_probe_code15_0307_mult_1724_OLD'
# backbone='resnet18'
# pretrain_path='/data1/1shared/lijun/ecg/E-Zero/checkpoints/resnet-18_new_preprocess_prmopt_0_ckpt.pth'

# cd ..
# cd /home/yanmingke/E-Zero/finetune/sub_script/code15
# bash sub_code15_mult.sh $taskname $backbone $pretrain_path

cd ..
cd finetune/sub_script/ptbxl
bash sub_ptbxl.sh $taskname $backbone $pretrain_path 
