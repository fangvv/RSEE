if [ "$#" -eq 2 ]
then
    DATA_DIR=$1
    MODEL_DIR=$2
else
    DATA_DIR=../HMDB51
    MODEL_DIR=logs_RSEE/BLSTM_spec_hmdb_pres50_t16_3m124_a.95e.05_ed5_ft15ds_lr.001_gu3_ep50/models/ckpt.best.pth.tar
fi

echo "Using data path: ${DATA_DIR} and model path: ${MODEL_DIR}"


# inference without BLSTM
# python -u main_base.py hmdb RGB --arch parallel_resnet50 --num_segments 16 --npb \
# --irte_joint \
# --exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 \
# --reso_list 224 168 112 84 --backbone_list parallel_resnet50 \
# --exp_decay \
# --init_tau 0.000001 --policy_also_backbone --policy_input_offset 3 \
# --batch-size 16 -j 16 --gpus 0 \
# --test_from ${MODEL_DIR} --data_dir ${DATA_DIR} \
# > log_test.log
# OUTPUT0=`cat log_test.log | tail -n 3`

# inference with full BLSTM
python -u main_base.py hmdb RGB --arch parallel_resnet50 --num_segments 10 --npb \
--irte_joint \
--exp_header X --ada_reso_skip --policy_backbone mobilenet_v2 \
--reso_list 224 168 112 84 --backbone_list parallel_resnet50 \
--exp_decay \
--init_tau 0.000001 --policy_also_backbone --policy_input_offset 3 \
--batch-size 16 -j 16 --gpus 0 \
--blstm --irte_final \
--test_from ${MODEL_DIR} --data_dir ${DATA_DIR} \
> log_test.log
OUTPUT0=`cat log_test.log | tail -n 3`