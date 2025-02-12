# =================================RSEE======================================
# train parallel_resnet50 backbone
python -u main_base.py hmdb RGB \
    --online_train \
    --irte \
    --arch parallel_resnet50 \
    --rescale_to 224 \
    --reso_list 224 168\
    --num_segments 10 \
    --gd 20 --dropout 0.5 --consensus_type=avg --npb --eval-freq=1 -j 36 --gpus 0 \
    --lr 0.01 \
    --print-freq 50 \
    --wd 1e-4 \
    --lr_steps 20 40 \
    --epochs 50 \
    --batch-size 32 \
    --exp_header hmdb_parallel_res50_t16_epo50_multiReso_lr.001 \
    --tune_from \
    logs_RSEE/RTENet_bk_ucf101_parallel_res50_t16_epo50_multiReso_lr.001/models/ckpt.best.pth.tar \
    --frm max \
    --data_dir ../HMDB51 --log_dir logs_RSEE > log_res50.log

# train baseline resnet50
# python -u main_base.py ucf101 RGB --baseline \
#     --arch resnet50 \
#     --rescale_to 224 \
#     --num_segments 10 \
#     --gd 20 --dropout 0.5 --consensus_type=avg --npb --eval-freq=1 -j 36 --gpus 0 \
#     --lr 0.001 \
#     --print-freq 50 \
#     --wd 1e-4 \
#     --lr_steps 20 40 \
#     --epochs 50 \
#     --batch-size 8 \
#     --exp_header baseline_ucf101_res50_t10_epo50__lr.001 \
#     --data_dir ../UCF101 --log_dir logs_RSEE > log_res50.log

# train baseline blstm
# python -u main_base.py ucf101 RGB \
#     --arch resnet50 --num_segments 10 --epochs 100 --batch-size 24 -j 16 \
#     --npb --gpus 0 --exp_header BLSTM_ucf101_res50_t10_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50 \
#     --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 \
#     --backbone_list resnet50 \
#     --print-freq 50 --lr 0.001 --lr_type step --lr_steps 20 40 \
#     --freeze_backbone \
#     --model_paths \
#     logs_RSEE/baseline_resnet50_bk_ucf101_res50_t10_epo50__lr.001/models/ckpt.best.pth.tar \
#     --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 0 \
#     --accuracy_weight 1 --efficency_weight 0 \
#     --uniform_loss_weight 0.0 --use_gflops_loss --random_seed 1007 \
#     --data_dir ../UCF101 --log_dir logs_RSEE \
#     --blstm --irte_final --baseline \
#     > log_joint.log

# train parallel_resnet50 backbone online (use multi blstm)
# python -u main_base.py hmdb RGB \
#     --irte --blstm --online_train --load_imagenet \
#     --arch parallel_resnet50 \
#     --rescale_to 224 \
#     --reso_list 224 168 112 84 \
#     --num_segments 10 \
#     --gd 20 --dropout 0.5 --consensus_type=avg --npb --eval-freq=1 -j 16 --gpus 0 \
#     --lr 0.001 \
#     --print-freq 50 \
#     --wd 1e-4 \
#     --lr_steps 20 40 \
#     --epochs 50 \
#     --batch-size 8 \
#     --exp_header online_BLSTM_hmdb_parallel_res50_t16_epo50_multiReso_lr.001 \
#     --data_dir ../HMDB51 --log_dir logs_RSEE >log_res50.log

# joint train policynet and parallel_resnet50
# python -u main_base.py hmdb RGB \
#     --irte_joint \
#     --arch parallel_resnet50 --num_segments 10 --lr 0.01 --epochs 50 --batch-size 32 -j 16 \
#     --npb --gpus 0 --exp_header hmdb_pres50_t10_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50 \
#     --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 \
#     --backbone_list parallel_resnet50 \
#     --print-freq 50 \
#     --freeze_backbone \
#     --model_paths \
#     logs_RSEE/parallel_resnet50_backbone_hmdb_parallel_res50_t16_epo100_multiReso_lr.001/models/ckpt.best.pth.tar \
#     --accuracy_weight 1 --efficency_weight 0.05 \
#     --uniform_loss_weight 0 --use_gflops_loss --random_seed 1007 \
#     --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 3 \
#     --data_dir ../HMDB51 --log_dir logs_RSEE > log_joint.log

# joint train policynet and BLSTM, freeze backbone
# python -u main_base.py hmdb RGB \
#     --irte_joint \
#     --arch parallel_resnet50 --num_segments 10 --epochs 50 --batch-size 12 -j 8 \
#     --npb --gpus 0 --exp_header BLSTM_hmdb_pres50_t10_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50 \
#     --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 \
#     --backbone_list parallel_resnet50 \
#     --print-freq 50 --lr 0.0005 --lr_type step --lr_steps 5 10 25 30 40 \
#     --model_paths \
#     logs_RSEE/RTE_L_BLSTM_hmdb_pres50_t10_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50/models/ckpt.best.pth.tar \
#     --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 3 \
#     --accuracy_weight 0.95 --efficency_weight 0.05 --TE 0.3 --use_gflops_loss  \
#     --random_seed 1007 \
#     --data_dir ../HMDB51 --log_dir logs_RSEE \
#     --blstm --irte_final \
#     > log_joint.log

# stan_lstm
# python -u main_base.py hmdb RGB \
#     --irte_joint \
#     --arch parallel_resnet50 --num_segments 10 --epochs 50 --batch-size 32 -j 16 \
#     --npb --gpus 0 --exp_header exit_w_0.01_BLSTM_hmdb_pres50_t16_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50 \
#     --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 \
#     --backbone_list parallel_resnet50 \
#     --print-freq 50 --lr 0.001 --lr_type step --lr_steps 20 40 \
#     --freeze_backbone \
#     --tune_from \
#     logs_RSEE/only_drs_hmdb_pres50_t16_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50/models/ckpt.best.pth.tar \
#     --accuracy_weight 1 --efficency_weight 0 \
#     --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 3 \
#     --uniform_loss_weight 0 --use_gflops_loss --random_seed 1007 \
#     --data_dir ../HMDB51 --log_dir logs_RSEE \
#     --stan_lstm --irte_final --exit_w 0 \
#     > log_joint.log

# train drs and stan_lstm from based on trained backbone
# python -u main_base.py hmdb RGB \
#     --irte_joint \
#     --arch parallel_resnet50 --num_segments 10 --epochs 50 --batch-size 32 -j 16 \
#     --npb --gpus 0 --exp_header exit_w_0.01_BLSTM_hmdb_pres50_t16_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50 \
#     --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 \
#     --backbone_list parallel_resnet50 \
#     --print-freq 50 --lr 0.001 --lr_type step --lr_steps 20 40 \
#     --freeze_backbone \
#     --model_paths \
#     logs_RSEE/parallel_resnet50_backbone_hmdb_parallel_res50_t16_epo100_multiReso_lr.001/models/ckpt.best.pth.tar \
#     --accuracy_weight 1 --efficency_weight 0 \
#     --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 3 \
#     --uniform_loss_weight 0 --use_gflops_loss --random_seed 1007 \
#     --data_dir ../HMDB51 --log_dir logs_RSEE \
#     --stan_lstm --irte_final --exit_w 0.1 \
#     > log_joint.log

# joint train policynet with frm func MaxPooling without BLTSM
# python -u main_base.py hmdb RGB \
#     --irte_joint \
#     --arch parallel_resnet50 --num_segments 16 --epochs 50 --batch-size 32 -j 16 \
#     --npb --gpus 0 --exp_header MaxPooling_hmdb_pres50_t16_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50 \
#     --ada_reso_skip --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 \
#     --backbone_list parallel_resnet50 \
#     --print-freq 50 --lr 0.001 --lr_type step --lr_steps 20 40 \
#     --freeze_backbone \
#     --tune_from \
#     logs_RSEE/only_drs_hmdb_pres50_t16_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50/models/ckpt.best.pth.tar \
#     --accuracy_weight 1 --efficency_weight 0 \
#     --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 3 \
#     --uniform_loss_weight 0 --use_gflops_loss --random_seed 1007 \
#     --data_dir ../HMDB51 --log_dir logs_RSEE \
#     --frm max --irte_final --exit_loss --exit_w 0.1 --fake_r\
#     > log_joint.log

# joint train backbone and BLSTM without drs
# python -u main_base.py hmdb RGB \
#     --irte_joint --load_imagenet --online_train --ada_reso_skip \
#     --arch parallel_resnet50 --num_segments 16 --epochs 50 --batch-size 4 -j 16 \
#     --npb --gpus 0 --exp_header BLSTM_hmdb_pres50_t16_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50 \
#     --policy_backbone mobilenet_v2 --reso_list 224 168 112 84 \
#     --backbone_list parallel_resnet50 \
#     --print-freq 50 --lr 0.001 --lr_type step --lr_steps 20 40 \
#     --accuracy_weight 0.95 --efficency_weight 0.05 \
#     --tune_from \
#     logs_RSEE/only_drs_hmdb_pres50_t16_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50/models/ckpt.best.pth.tar \
#     --exp_decay --init_tau 5 --policy_also_backbone --policy_input_offset 3 \
#     --uniform_loss_weight 3.0 --use_gflops_loss --random_seed 1007 \
#     --data_dir ../HMDB51 --log_dir logs_RSEE \
#     --blstm --irte_final --exit_w 0.1 \
#     > log_joint.log